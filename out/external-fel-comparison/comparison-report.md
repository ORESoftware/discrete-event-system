# External FEL Comparison

| Domain | Scenario | Engine | Status | Checks | Input | Output |
| --- | --- | --- | --- | ---: | --- | --- |
| traffic | five-intersection-scheduled-trips | source/sink input | passed | 108/108 | out/external-fel-comparison/traffic-shared-input.json |  |
| traffic | five-intersection-scheduled-trips | Python traffic FEL | passed | 7/7 | out/external-fel-comparison/traffic-shared-input.json | out/external-fel-comparison/python-traffic-fel.json |
| traffic | five-intersection-scheduled-trips | SUMO | passed | 7/7 | out/external-fel-comparison/traffic-shared-input.json | out/external-fel-comparison/sumo.json |
| computer-network | small-enterprise | Python computer-network FEL | passed | 13/13 | out/external-fel-comparison/computer-network-small-enterprise.json | out/external-fel-comparison/computer-network-small-enterprise-python-fel-reference.json |
| computer-network | bottleneck-lab | Python computer-network FEL | passed | 13/13 | out/external-fel-comparison/computer-network-bottleneck-lab.json | out/external-fel-comparison/computer-network-bottleneck-lab-python-fel-reference.json |
