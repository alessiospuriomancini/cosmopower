import cosmopower as cp
import pytest

def test_parser():
    parser = cp.YAMLParser("example.yaml")
    networks = parser.restore_networks()
    print(networks)

    for quantity in networks:
        network = networks[quantity]

        print(quantity, network.modes)

