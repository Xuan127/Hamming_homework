import networkx as nx
import matplotlib.pyplot as plt
from deprecated.helper_structs import ConversationState

class ConversationGraph:
    def __init__(self):
        # Initialize a directed graph
        self.graph = nx.DiGraph()
        self.add_node('start', '')
        self.information_database = []

    def add_node(self, node_id, state, history=[]):
        node_id = self.wrap_text(node_id, 15)
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, state=state, history=history)
        else:
            print(f"Node '{node_id}' already exists.")

    def add_edge(self, from_node, to_node, condition):
        from_node = self.wrap_text(from_node, 15)
        to_node = self.wrap_text(to_node, 15)
        condition = self.wrap_text(condition, 15)
        if self.graph.has_node(from_node) and self.graph.has_node(to_node):
            self.graph.add_edge(from_node, to_node, condition=condition)
        else:
            raise ValueError("Both nodes must exist in the graph before adding an edge.")

    def add_node_with_edge(self, from_node, to_node, to_state, condition, history=[]):
        self.add_node(to_node, to_state, history)
        self.add_edge(from_node, to_node, condition)
        
    def get_next_state(self, current_node, condition):
        # Iterate through all outgoing edges from current_node
        for _, to_node, data in self.graph.out_edges(current_node, data=True):
            if data.get('condition') == condition:
                return to_node
        return None
    
    def get_node_state(self, node_id):
        node_id = self.wrap_text(node_id, 15)
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id].get('state')
        else:
            raise ValueError("Node does not exist in the graph.")
    
    def get_history(self, node_id):
        node_id = self.wrap_text(node_id, 15)
        if self.graph.has_node(node_id):
            return self.graph.nodes[node_id].get('history', [])
        else:
            raise ValueError("Node does not exist in the graph.")

    def wrap_text(self, text, max_length):
        if len(text) > max_length:
            last_newline = 0
            chars = list(text)
            for i in range(len(chars)):
                if chars[i] == ' ' and i - last_newline >= max_length:
                    chars[i] = '\n'
                    last_newline = i
            text = ''.join(chars)
        return text
    
    def visualize_graph(self):
        # Use hierarchical layout for downward direction
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        # Adjust y-coordinates to ensure downward flow
        y_max = max(coord[1] for coord in pos.values())
        y_min = min(coord[1] for coord in pos.values())
        for node in pos:
            level = nx.shortest_path_length(self.graph, list(self.graph.nodes())[0], node)
            pos[node] = (pos[node][0], y_max - (level * (y_max - y_min) / (len(self.graph.nodes()) - 1)))
    
        states = nx.get_node_attributes(self.graph, 'state')
        conditions = nx.get_edge_attributes(self.graph, 'condition')
    
        # Draw nodes with different shapes based on state
        question_nodes = [node for node, attr in states.items() if attr == ConversationState.QUESTION or attr == ConversationState.ACTION_REQUEST \
                          or attr == ConversationState.CLARIFICATION or attr == ConversationState.CONFIRMATION]
        action_nodes = [node for node, attr in states.items() if attr == ConversationState.ACTION]
        end_nodes = [node for node, attr in states.items() if attr == ConversationState.END]
        information_nodes = [node for node, attr in states.items() if attr == ConversationState.INFORMATION]
    
        # Draw question nodes as rectangles
        nx.draw_networkx_nodes(self.graph, pos, 
                             nodelist=question_nodes,
                             node_color='lightblue',
                             node_size=2000,
                             node_shape='D')  # 's' for square
    
        # Draw action nodes as circles
        nx.draw_networkx_nodes(self.graph, pos,
                             nodelist=action_nodes,
                             node_color='lightgreen',
                             node_size=2000,
                             node_shape='o')  # 'o' for circle
    
        # Draw end nodes as triangles
        nx.draw_networkx_nodes(self.graph, pos,
                             nodelist=end_nodes,
                             node_color='lightcoral',
                             node_size=2000,
                             node_shape='^')  # '^' for triangle
    
        # Draw information nodes as diamonds
        nx.draw_networkx_nodes(self.graph, pos,
                             nodelist=information_nodes,
                             node_color='yellow',
                             node_size=2000,
                             node_shape='s')  # 'D' for diamond
    
        # Draw edges and labels
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Adjust label positions - only for end nodes
        label_pos = pos.copy()
        for p in label_pos:
            if p in end_nodes:  # Only move labels for end nodes
                label_pos[p] = (label_pos[p][0], label_pos[p][1] - 0.02)  # Shift labels down slightly
            
        nx.draw_networkx_labels(self.graph, label_pos, font_size=7, font_weight='bold')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=conditions, font_color='red', font_size=7)
        
        plt.axis('off')
        plt.savefig('conversation_graph.png')

if __name__ == "__main__":
    graph = ConversationGraph()
    graph.add_node_with_edge("start", "middle", ConversationState.ACTION, "conditionnalfjd asdfhausdfhsdfhasj dfhaskjjdfhafdh dasjdfhasjkf hjkasfhaksfhajsfdhaks jdfhasdfhjkasfa fdsafafasdfasdf")
    graph.add_node_with_edge("middle", "end", ConversationState.END, "condition")
    graph.add_node_with_edge("end", "another end", ConversationState.INFORMATION, "condition")
    graph.visualize_graph()
