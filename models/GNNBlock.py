import torch
import torch_geometric.nn as pygnn


class GNNBlock(torch.nn.Module):

    def __init__(
        self, 
        gnn_class="GATv2Conv", 
        in_channels=768, # BERT encoder output size
        lin_in_channels=768,
        out_channels=64, 
        add_self_loops=True,
        gnn_kwargs=dict(heads=1),
        lin_kwargs={},
        non_linearity=torch.nn.ReLU(), # without an inplace=True arg
    ):

        assert gnn_class in ("GATv2Conv", "GATConv", "GCNConv")

        super(GNNBlock, self).__init__()
        self.gnn = pygnn.__dict__[gnn_class](
            in_channels, out_channels, add_self_loops=add_self_loops, **gnn_kwargs,
        )

        self.linear = pygnn.Linear(
            in_channels=lin_in_channels, out_channels=out_channels, **lin_kwargs,
        )

        self.non_linearity = non_linearity

    def forward(self, node, edge_index):
        node = self.gnn(node, edge_index) + self.linear(node)
        if self.non_linearity is not None:
            node = self.non_linearity(node)
        return node