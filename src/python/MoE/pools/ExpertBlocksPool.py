import asyncio
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple


@dataclass
class ExpertBlockMetaData:
    is_alive: bool  # Is alive flag.
    alive_event: asyncio.Event  # If this expert block is alive, this event is set, otherwise it is cleared.
    experts_block_index: int  # Index of expert block.
    expert_indexes: List[int]  # List of expert indexes in this block.


class ExpertBlocksPool:
    def __init__(
        self,
        expert_blocks: List[List[int]],  # List of expert indexes in each of the blocks.
    ) -> None:
        # Number of expert blocks.
        self.number_of_expert_blocks = len(expert_blocks)

        # Expert blocks metadata and pool of their indexes.
        self.expert_blocks_metadata: List[ExpertBlockMetaData] = []
        self.expert_block_indexes_pool: asyncio.Queue = asyncio.Queue()
        for experts_block_index in range(self.number_of_expert_blocks):
            self.expert_blocks_metadata.append(
                ExpertBlockMetaData(
                    True,
                    alive_event=asyncio.Event(),
                    experts_block_index=experts_block_index,
                    expert_indexes=expert_blocks[experts_block_index],
                )
            )
            self.expert_blocks_metadata[-1].alive_event.set()
            self.expert_block_indexes_pool.put_nowait(experts_block_index)

    async def aquire(self) -> Tuple[int, ExpertBlockMetaData]:
        while True:
            try:
                experts_block_index: int = await self.expert_block_indexes_pool.get()
                metadata = self.expert_blocks_metadata[experts_block_index]
                if metadata.alive_event.is_set():
                    return experts_block_index, metadata
            except (asyncio.QueueEmpty, AttributeError):
                raise RuntimeError("Expert blocks async queue is corrupted.")

    def release(self, experts_block_index: int):
        if self.expert_blocks_metadata[experts_block_index].alive_event.is_set():
            self.expert_block_indexes_pool.put_nowait(experts_block_index)

    def mark_alive(self, experts_block_index: int) -> None:
        expert_block_metadata = self.expert_blocks_metadata[experts_block_index]
        if not expert_block_metadata.alive_event.is_set():
            expert_block_metadata.alive_event.set()
            self.expert_block_indexes_pool.put_nowait(experts_block_index)
        expert_block_metadata.is_alive = True

    def mark_dead(self, experts_block_index: int) -> None:
        expert_block_metadata = self.expert_blocks_metadata[experts_block_index]
        if expert_block_metadata.alive_event.is_set():
            expert_block_metadata.alive_event.clear()
        expert_block_metadata.is_alive = False

    def close(self) -> None:
        while not self.expert_block_indexes_pool.empty():
            try:
                experts_block_index: int = self.expert_block_indexes_pool.get_nowait()
                expert_block_metadata = self.expert_blocks_metadata[experts_block_index]
                expert_block_metadata.alive_event.clear()
            except Exception:
                break
