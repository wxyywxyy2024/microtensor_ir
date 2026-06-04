# src/optimizer.py
from typing import Dict, Tuple
from .frontend import Graph

class LivenessAnalyzer:
    @staticmethod
    def analyze(graph: Graph) -> Dict[str, Tuple[int, int]]:
        """
        Returns a dictionary mapping each Tensor Value name to its explicit 
        lifespan interval: (birth_instruction_idx, death_instruction_idx)
        """
        lifespans = {}
        
        # 1. Inputs are alive from the absolute beginning (Line 0)
        for inp in graph.inputs:
            lifespans[inp.name] = [0, 0]

        # 2. Iterate through the linearized execution sequence
        for idx, node in enumerate(graph.nodes, start=1):
            # The instruction index represents the current time step
            
            # Record birth of outputs created by this operation
            for out in node.outputs:
                lifespans[out.name] = [idx, idx]
                
            # Update the death cycle of inputs (extending their lifespan to the current instruction)
            for inp in node.inputs:
                if inp.name in lifespans:
                    lifespans[inp.name][1] = idx
                    
        # 3. Graph outputs must remain alive through the very end of execution
        end_idx = len(graph.nodes)
        for out in graph.outputs:
            if out.name in lifespans:
                lifespans[out.name][1] = end_idx

        # Convert list bounds to a clean tuple mapping
        return {name: (bounds[0], bounds[1]) for name, bounds in lifespans.items()}