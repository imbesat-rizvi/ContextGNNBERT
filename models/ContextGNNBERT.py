import torch
import torch_geometric as pyg
from transformers.modeling_outputs import SequenceClassifierOutput

from .EncoderBERT import EncoderBERT
from .GNNBlock import GNNBlock


class ContextGNNBERT(EncoderBERT):

    def __init__(
        self, 
        encoder, 
        num_labels=2,
        trainable_encoder=False,
        num_layers=2,
        gnn_class="GATv2Conv", 
        hidden_channels=64,
        gnn_kwargs=dict(heads=1),
        gnn_block_dropout=0.2,
        gnn_lin_kwargs={},
        non_linearity="ReLU",
    ):
        
        super(ContextGNNBERT, self).__init__(encoder, num_labels, trainable_encoder)

        self.gnn = self.create_gnn(
            num_labels=num_labels,
            num_layers=num_layers,
            gnn_class=gnn_class, 
            hidden_channels=hidden_channels,
            gnn_kwargs=gnn_kwargs,
            gnn_block_dropout=gnn_block_dropout,
            lin_kwargs=gnn_lin_kwargs,
            non_linearity=non_linearity,
        )

        self.hetero_gnn = False

    
    def forward(
        self, input_ids, attention_mask, token_type_ids, context_mask, labels=None
    ):
        r"""placeholder labels argument is provided for the compatibility with
        the huggingface dataset and Trainer api's compute_metrics."""
        
        seq_out, pooled_out = self.encoder(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        graph = self.create_context_graph(
            seq_out, pooled_out, token_type_ids, context_mask
        )

        self.to_hetero_gnn(graph.metadata())
        for module in self.gnn:
            graph.node_dict = module(graph.node_dict, graph.edge_index_dict)
        
        output = graph.node_dict["pooled_out"]
        return SequenceClassifierOutput(logits=output)


    def create_gnn(
        self,
        num_labels=2,
        num_layers=2,
        gnn_class="GATv2Conv", 
        hidden_channels=64,
        gnn_kwargs=dict(heads=1),
        gnn_block_dropout=0.2,
        lin_kwargs={},
        non_linearity="ReLU",
    ):

        in_channels, out_channels = self.get_in_out_channels(
            num_labels=num_labels,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
        )

        gnn = torch.nn.ModuleList()

        for i, in_ch, out_ch in zip(
            range(1, num_layers+1), in_channels, out_channels
        ):

            gnn_block = GNNBlock(
                gnn_class, 
                in_channels=in_ch,
                lin_in_channels=in_ch,
                out_channels=out_ch,
                add_self_loops=False, # add self loop in the data itself
                gnn_kwargs=gnn_kwargs,
                lin_kwargs=lin_kwargs,
                non_linearity=non_linearity if i < num_layers else None,
                dropout=gnn_block_dropout if i < num_layers else None,
            )

            gnn.append(gnn_block)

        return gnn

    
    def to_hetero_gnn(self, metadata, aggr="sum"):
        # convert gnn to hetero on getting 1st batch and omit this step later
        if not self.hetero_gnn:
            self.gnn = torch.nn.ModuleList([
                pyg.nn.to_hetero(m, metadata, aggr=aggr) for m in self.gnn
            ])
            self.hetero_gnn = True

    
    def create_context_graph(
        self, 
        seq_out, 
        pooled_out, 
        token_type_ids,
        context_mask,
        undirected=True,
        self_loops=True,
        normalize_feats=False,
    ):
        r"""The graph being created is a star graph with [CLS] pooler output as 
        its center"""
        graph = pyg.data.HeteroData()
        
        # node type of central [CLS] token in the graph
        graph["pooled_out"].node = pooled_out # B x D
        
        # leaf nodes of the graph from the sequence
        # reshape nodes from B x T x D to B*T x D
        graph["seq1_elm"].node = seq_out.view((-1,seq_out.shape[-1]))
        
        
        paired_seq = torch.any(token_type_ids)
        if not paired_seq:
            graph["pooled_out", "edge", "seq1_elm"].edge_index = (
                self.form_edge_idcs(context_mask)
            )
        
        else:
            # form another type of leaf nodes of the graph from 2nd seq.
            # can point to the same node feature matrix, as seq_out
            # contains values from both the sequences i.e. token_type_ids.
            graph["seq2_elm"].node = graph["seq1_elm"].node

            graph["pooled_out", "edge", "seq1_elm"].edge_index = (
                self.form_edge_idcs(context_mask, token_type_ids, seq_id=0)
            )

            graph["pooled_out", "edge", "seq2_elm"].edge_index = (
                self.form_edge_idcs(context_mask, token_type_ids, seq_id=1)
            )

        if undirected:
            graph = pyg.transforms.ToUndirected()(graph)
        if self_loops:
            graph = pyg.transforms.AddSelfLoops()(graph)
        if normalize_feats:
            graph = pyg.transforms.NormalizeFeatures()(graph)

        return graph

    
    def form_edge_idcs(
        self, context_mask, token_type_ids=None, seq_id=0, pooled_out_first=True
    ):
        seq_len = context_mask.shape[1]
        if token_type_ids is not None:
            context_mask = (context_mask*(token_type_ids==seq_id)).view(-1)
        else:
            context_mask = context_mask.view(-1)

        # edge indices for sequences, should be of shape 2 x num_edges
        context_node_ids = torch.where(context_mask)[0]
        pooled_out_node_ids = torch.div(
            context_node_ids, seq_len, rounding_mode="floor"
        )

        if pooled_out_first:
            edge_idcs = torch.vstack((pooled_out_node_ids, context_node_ids))
        else:
            edge_idcs = torch.vstack((context_node_ids, pooled_out_node_ids))

        return edge_idcs