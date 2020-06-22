REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .dcg_controller import DeepCoordinationGraphMAC
REGISTRY["dcg_mac"] = DeepCoordinationGraphMAC

from .dcg_noshare_controller import DCGnoshareMAC
REGISTRY["dcg_noshare_mac"] = DCGnoshareMAC

from .cg_controller import SimpleCoordionationGraphMAC
REGISTRY["cg_mac"] = SimpleCoordionationGraphMAC

from .low_rank_controller import LowRankMAC
REGISTRY["low_rank_q"] = LowRankMAC

