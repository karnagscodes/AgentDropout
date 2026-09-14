import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import warnings
import asyncio

from AgentDropout.utils import instrument


class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.
        
    Methods:
        add_predecessor(operation): 
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation): 
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs): 
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs): 
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(self, 
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="", 
                 llm_name:str = "",
                 ):
        """
        Initializes a new Node instance.
        """
        self.id:str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {'inputs':[],'outputs':[],'raw_inputs':[]}        

    @property
    def node_name(self):
        return self.__class__.__name__
    
    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st =='temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st =='temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []        
    
    def update_memory(self):
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    def get_spatial_info(self)->Dict[str,Dict]:
        """ Return a dict that maps id to info. """
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_outputs = predecessor.outputs
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                spatial_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return spatial_info

    def get_temporal_info(self)->Dict[str,Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                temporal_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}
        
        return temporal_info
    
    def execute(self, input:Any, **kwargs):
        self.outputs = []
        spatial_info:Dict[str,Dict] = self.get_spatial_info()
        temporal_info:Dict[str,Dict] = self.get_temporal_info()
        results = [self._execute(input, spatial_info, temporal_info, **kwargs)]

        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs


    def _render_len(self, raw_inputs, spatial_info, temporal_info) -> int:
        """Token length of the prompt this node's own renderer would build."""
        rendered = self._process_inputs(raw_inputs, spatial_info, temporal_info)
        if not isinstance(rendered, (tuple, list)):
            rendered = (rendered,)
        return sum(instrument.count_tokens(part) for part in rendered if isinstance(part, str))

    def _log_edge_costs(self, raw_inputs, spatial_info, temporal_info):
        """Attribute prompt tokens to the predecessors that contributed them.

        Rendering with one predecessor and subtracting the empty render gives
        the segment cost as this node actually formats it, which matters
        because roles differ: Math Solver receives an extracted number while
        every other role receives the full verbatim output.
        """
        try:
            base = self._render_len(raw_inputs, {}, {})
            full = self._render_len(raw_inputs, spatial_info, temporal_info)
        except Exception as exc:
            instrument.log("edges_error", node_id=self.id,
                           node_name=self.node_name, error=repr(exc))
            return

        for kind, info in (("spatial", spatial_info), ("temporal", temporal_info)):
            for sender_id, meta in info.items():
                only = {sender_id: meta}
                try:
                    with_one = (self._render_len(raw_inputs, only, {}) if kind == "spatial"
                                else self._render_len(raw_inputs, {}, only))
                except Exception as exc:
                    instrument.log("edges_error", node_id=self.id, sender_id=sender_id,
                                   error=repr(exc))
                    continue
                output = meta.get("output", "") if isinstance(meta, dict) else ""
                instrument.log(
                    "edges",
                    edge_type=kind,
                    sender_id=sender_id,
                    sender_role=meta.get("role", "") if isinstance(meta, dict) else "",
                    receiver_id=self.id,
                    receiver_name=self.node_name,
                    receiver_role=self.role,
                    rendered_segment_tokens=with_one - base,
                    sender_output_tokens=instrument.count_tokens(output),
                    sender_output=output,
                )

        instrument.log(
            "prompt_build",
            node_id=self.id,
            node_name=self.node_name,
            role=self.role,
            base_prompt_tokens=base,
            full_prompt_tokens=full,
            n_spatial=len(spatial_info),
            n_temporal=len(temporal_info),
        )

    async def async_execute(self, input:Any, **kwargs):

        self.outputs = []
        spatial_info:Dict[str,Any] = self.get_spatial_info()
        temporal_info:Dict[str,Any] = self.get_temporal_info()
        # print(temporal_info)
        # Set identity, then create the task. create_task snapshots the context,
        # so concurrently executing nodes each see their own values. Do not move
        # these lines or insert an await before create_task.
        instrument.CTX_NODE_ID.set(self.id)
        instrument.CTX_NODE_NAME.set(self.node_name)
        self._log_edge_costs(input, spatial_info, temporal_info)
        # Set role after rendering: some agents (FinalRefer) assign self.role
        # inside _process_inputs, so reading it earlier logs an empty string.
        instrument.CTX_ROLE.set(self.role)
        tasks = [asyncio.create_task(self._async_execute(input, spatial_info, temporal_info, **kwargs))]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs
               
    @abstractmethod
    def _execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
