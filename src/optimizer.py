# src/optimizer.py
from typing import Dict, Tuple
from .frontend import Graph
import math

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

class StaticMemoryAllocator:
    @staticmethod
    def calculate_bytes(shape: tuple, dtype: str) -> int:
        """Calculates total bytes needed for a tensor shape assuming float32 for now."""
        element_size = 4 if dtype == "float32" else 2 # 4 bytes for float32
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        return total_elements * element_size

    @classmethod
    def allocate(cls, graph) -> dict:
        """
        Greedy Lifetime Interval Allocation.
        Return a dict of tensor names mapped to their starting byte offset
        in the shared global memory pool, alongside the total pool size required.
        """
        # 1. Get the lifespan intervals we built earlier
        intervals = LivenessAnalyzer.analyze(graph)

        # 2. Separate variables that need allocation
        # Note: We skip model inputs (%t1, %t2, %t3) because they are provided by the user
        # We only allocate static memory for internal intermediate scratch buffers
        internal_tensors = [t for t in graph.nodes]

        # Dictionary to store calculated byte offsets
        offsets = {}
        peak_memory_pool = 0

        # 3. Process allocations chronologically based on birth line
        for node in internal_tensors:
            out_tensor = node.outputs[0]
            t_name = out_tensor.name
            t_size = cls.calculate_bytes(out_tensor.shape, out_tensor.dtype)
            birth, death = intervals[t_name]

            # Find a safe offset for this tensor
            candidate_offset = 0
            allocated = False

            while not allocated:
                overlap_found = False
                # Check against everything we've already allocated
                for allocated_name, alloc_offset in offsets.items():
                    alloc_size = cls.calculate_bytes(
                        next(t.outputs[0] for t in internal_tensors if t.outputs[0].name == allocated_name).shape,
                        "float32"
                    )

                    # Do their lifespans intersect?
                    a_birth, a_death = intervals[allocated_name]
                    lifespans_overlap = not (death < a_birth or birth > a_death)

                    # Do their proposed memory blocks intersect?
                    mem_overlap = not (candidate_offset + t_size <= alloc_offset or candidate_offset >= alloc_offset + alloc_size)

                    if lifespans_overlap and mem_overlap:
                        # Conflict! Bump candidate offset past the conflicting block and try again
                        candidate_offset = alloc_offset + alloc_size
                        overlap_found = True
                        break

                if not overlap_found:
                    allocated = True
        
            # Assign the safe offset
            offsets[t_name] = candidate_offset
            peak_memory_pool = max(peak_memory_pool, candidate_offset + t_size)

        return offsets, peak_memory_pool

