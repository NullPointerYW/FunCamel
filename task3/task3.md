<h1>workforce机制</h1>
<h2>workforce类</h2>
以给出的示例为例子

```python 
# 创建一个 Workforce 实例
workforce = Workforce(description="旅游攻略制作与评估工作组",new_worker_agent_kwargs={'model':model},coordinator_agent_kwargs={'model':model},task_agent_kwargs={'model':model})

```
点开`workforce.py`,可以看到`wordforce`类继承的是`Basenode`类，用于管理多个工作节点，协调任务的分配和处理<br>

`Base`类主要包含以下`attribute`:

```python 
    def __init__(self, description: str) -> None:
        self.node_id = str(id(self)) #node_id 节点唯一的标识符
        self.description = description # description 节点的描述信息
        self._channel: TaskChannel = TaskChannel() #任务通道，用于节点间的通信
        self._running = False # 表示节点是否正在运行的标志
```

`Workforce`类主要包含

```python 
def __init__(
        self,
        description: str,
        children: Optional[List[BaseNode]] = None,
        coordinator_agent_kwargs: Optional[Dict] = None,
        task_agent_kwargs: Optional[Dict] = None,
        new_worker_agent_kwargs: Optional[Dict] = None,
    ) -> None:
       super().__init__(description)
        self._child_listening_tasks: Deque[asyncio.Task] = deque()
        self._children = children or []
        self.new_worker_agent_kwargs = new_worker_agent_kwargs

        coord_agent_sys_msg = BaseMessage.make_assistant_message(
            role_name="Workforce Manager",
            content="You are coordinating a group of workers. A worker can be "
            "a group of agents or a single agent. Each worker is "
            "created to solve a specific kind of task. Your job "
            "includes assigning tasks to a existing worker, creating "
            "a new worker for a task, etc.",
        )
        self.coordinator_agent = ChatAgent(
            coord_agent_sys_msg, **(coordinator_agent_kwargs or {})
        )

        task_sys_msg = BaseMessage.make_assistant_message(
            role_name="Task Planner",
            content="You are going to compose and decompose tasks.",
        )
        self.task_agent = ChatAgent(task_sys_msg, **(task_agent_kwargs or {}))

        # If there is one, will set by the workforce class wrapping this
        self._task: Optional[Task] = None
        self._pending_tasks: Deque[Task] = deque()
```

`new_worker_agent_kwargs={'model':model}`为一个包含模型对象的词典，通过`Optional[Dict]`传入实例，`description`以`str`类型传入<br>

<h2>架构设计</h2>

在教程中提到<br>

>工作节点由内部的协调智能体(coordinator agent)管理，协调智能体根据工作节点的描述及其工具集来分配任务<br>

在`def __init__`中可以看到

```python
coord_agent_sys_msg = BaseMessage.make_assistant_message(
            role_name="Workforce Manager",
            content="You are coordinating a group of workers. A worker can be "
            "a group of agents or a single agent. Each worker is "
            "created to solve a specific kind of task. Your job "
            "includes assigning tasks to a existing worker, creating "
            "a new worker for a task, etc.",
        )
        self.coordinator_agent = ChatAgent(
            coord_agent_sys_msg, **(coordinator_agent_kwargs or {})
        )


```

协调智能体也是`ChatAgent`类，并且通过`role_name`和`content`来控制行为，通过修改content或许可以做提升<be>

任务智能体也同理
```python
task_sys_msg = BaseMessage.make_assistant_message(
            role_name="Task Planner",
            content="You are going to compose and decompose tasks.",
        )
        self.task_agent = ChatAgent(task_sys_msg, **(task_agent_kwargs or {}))


```

<h2>任务拆分</h2>

一个task是如何拆分成subtask的？在`workforce`类中的`process_task`,`_decompose_task`方法中,可以看到

```python

def _decompose_task(self, task: Task) -> List[Task]:
        r"""Decompose the task into subtasks. This method will also set the
        relationship between the task and its subtasks.

        Returns:
            List[Task]: The subtasks.
        """
        decompose_prompt = WF_TASK_DECOMPOSE_PROMPT.format(
            content=task.content,
            child_nodes_info=self._get_child_nodes_info(),
            additional_info=task.additional_info,
        )
        self.task_agent.reset()
        subtasks = task.decompose(self.task_agent, decompose_prompt)
        task.subtasks = subtasks
        for subtask in subtasks:
            subtask.parent = task

        return subtasks

def process_task(self,task:Task) -> Task 
subtasks = self._decompose_task(task)
```
这里调用了task类中的`decompose`方法，我们点开可以看到
```python
def decompose(
        self,
        agent: ChatAgent,
        prompt: Optional[str] = None,
        task_parser: Callable[[str, str], List["Task"]] = parse_response,
    ) -> List["Task"]:
        r"""Decompose a task to a list of sub-tasks. It can be used for data
        generation and planner of agent.

        Args:
            agent (ChatAgent): An agent that used to decompose the task.
            prompt (str, optional): A prompt to decompose the task. If not
                provided, the default prompt will be used.
            task_parser (Callable[[str, str], List[Task]], optional): A
                function to extract Task from response. If not provided,
                the default parse_response will be used.

        Returns:
            List[Task]: A list of tasks which are :obj:`Task` instances.
        """

        role_name = agent.role_name
        content = prompt or TASK_DECOMPOSE_PROMPT.format(
            role_name=role_name,
            content=self.content,
        )
        msg = BaseMessage.make_user_message(
            role_name=role_name, content=content
        )
        response = agent.step(msg)
        tasks = task_parser(response.msg.content, self.id)
        for task in tasks:
            task.additional_info = self.additional_info
        return tasks

```

我们将`workforce`中已经设定好的，包含着`task_sys_masg`的`self.task_agent`和设置好的`WF_TASK_DECOMPOSE_PROMPT`(通过format方法替换占位符)得到的`decompose_prompt`作为两个变量带入到`decompose`方法，并返回`list[task]`为子任务

<h2>方法链(fluent Interface)</h2>

方法链(Method Chaining)，也称为流畅借口(Fluent Interface),是一种编程风格，允许通过连续调用对象的方法来实现复杂的操作。每个方法都返回对象本身(`self`),从而可以继续调用其他方法,以**workforce**的`add_single_agent_worker`为例


```python
@check_if_running(False)
    def add_single_agent_worker(
        self, description: str, worker: ChatAgent
    ) -> Workforce:
        worker_node = SingleAgentWorker(description, worker)
        self._children.append(worker_node)
        return self

# 添加工作节点
workforce.add_single_agent_worker(
    "负责搜索目的地相关信息",
    worker=search_agent
).add_single_agent_worker(
    "负责制定详细行程规划",
    worker=planner_agent
).add_single_agent_worker(
    "负责从游客角度评估行程",
    worker=reviewer_agent
)
```
每一次`add_single_agent_worker`都会添加一个工作节点，返回`self`即`wordforce`本身，`workforce.add_single_agent_worker`继承`workforce`，又可以调用`add_single_agent_worker`增加节点

<h2> worker 类 </h2>

worker本质是`ChatAgent`类的实例:

```python
@track_agent(name="ChatAgent")
class ChatAgent(BaseAgent):
def __init__(
        self,
        system_message: Optional[Union[BaseMessage, str]] = None,
        model: Optional[
            Union[BaseModelBackend, List[BaseModelBackend]]
        ] = None,
        memory: Optional[AgentMemory] = None,
        message_window_size: Optional[int] = None,
        token_limit: Optional[int] = None,
        output_language: Optional[str] = None,
        tools: Optional[List[Union[FunctionTool, Callable]]] = None,
        external_tools: Optional[List[Union[FunctionTool, Callable]]] = None,
        response_terminators: Optional[List[ResponseTerminator]] = None,
        scheduling_strategy: str = "round_robin",
        single_iteration: bool = False,
    ) -> None:
```

每一个worker定义了`system_message`,`model`,`output`,如果需要更加详细可以根据`ChatAgent`类自行添加


将worker添加至工作节点
```python 
# 添加一个执行网页搜索的Agent
workforce.add_single_agent_worker(
    "一个能够执行网页搜索的Agent",    worker=search_agent,
)
```
其中在`workforce`中的
```python
worker_node = SingleAgentWorker(description,worker)

```
通过传递参数直接构造的方式构造`SingleAgentWorker`类的`worker_node`实例