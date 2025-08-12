from genesis import gs
from genesis.engine.entities import RigidEntity
import torch

from repairs_components.geometry.fasteners import Fastener


def get_links_idx_from_names(names: list[str], gs_entities: dict[str, RigidEntity]):
    return torch.tensor(
        [gs_entities[name].base_link.idx for name in names], dtype=torch.int32
    )


# hmmm, this should be cached.
def get_links_idx_from_fastener_ids(
    fastener_ids: torch.Tensor, gs_entities: dict[str, RigidEntity]
):
    "Util to get link indices from fastener ids."
    # Minimize lookups into gs_entities by querying only for unique fasteners,
    # then reconstruct the full index list via inverse indices.
    unique_ids, inverse = torch.unique(fastener_ids, return_inverse=True)
    inverse_cpu = inverse.cpu()

    unique_names = [
        Fastener.fastener_name_in_simulation(fid.item()) for fid in unique_ids
    ]
    unique_link_idxs = torch.tensor(
        [gs_entities[name].base_link.idx for name in unique_names],
        dtype=torch.int32,
    )
    return unique_link_idxs[inverse_cpu].to(fastener_ids.device)
