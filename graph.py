import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NeuralNetworkGraphApp:
    def __init__(self, master, model_data, tokenizer_data):
        self.master = master
        self.model_data = model_data
        self.tokenizer_data = tokenizer_data

        self.G = self.create_graph()
        
        self.master.title("Neural Network Graph Visualizer")

        self.label = tk.Label(master, text="Select a neuron or token:")
        self.label.pack(pady=10)

        self.dropdown = ttk.Combobox(master, values=list(self.G.nodes()))
        self.dropdown.pack(pady=5)

        self.show_button = tk.Button(master, text="Show Connections", command=self.show_connections)
        self.show_button.pack(pady=10)

        self.save_button = tk.Button(master, text="Save Graph as Image", command=self.save_graph)
        self.save_button.pack(pady=5)

        self.canvas = None

    def create_graph(self):
        G = nx.DiGraph()
        logging.info("Creating graph...")

        # Add nodes for tokens from tokenizer data
        for token in self.tokenizer_data.keys():
            G.add_node(token, type='token')

        logging.info(f"Added {len(self.tokenizer_data)} tokens to the graph.")

        # Add nodes for model layers and their neurons
        for layer_index, layer in enumerate(self.model_data['layers']):
            for neuron_index in range(len(layer['weights'])):
                neuron_name = f'Layer {layer_index + 1} Neuron {neuron_index + 1}'
                G.add_node(neuron_name, type='neuron')

                # Connect the neuron to the corresponding token outputs
                for token_index in range(len(layer['weights'][neuron_index])):
                    token = list(self.tokenizer_data.keys())[token_index]
                    weight = layer['weights'][neuron_index][token_index]
                    if weight != 0:  # Only add edges for non-zero weights
                        G.add_edge(neuron_name, token, weight=weight)

                logging.info(f"Added connections from {neuron_name} to tokens.")

        logging.info("Graph creation completed.")
        return G

    def show_connections(self):
        selected_node = self.dropdown.get()
        if selected_node:
            self.plot_graph(selected_node)

    def plot_graph(self, selected_node):
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(self.G, seed=42)  # Positions for all nodes

        # Draw the selected node and its neighbors
        neighbors = list(self.G.neighbors(selected_node))
        neighbors.append(selected_node)  # Include the selected node

        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, nodelist=neighbors, node_size=300, node_color='lightcoral', label='Connected Nodes')
        
        # Draw edges with weights
        edges = self.G.edges(neighbors)
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.G.edges(data=True) if u in neighbors and v in neighbors}
        nx.draw_networkx_edges(self.G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=10)
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        # Draw labels
        labels = {node: node for node in neighbors}
        nx.draw_networkx_labels(self.G, pos, labels, font_size=10)

        plt.title(f"Connections for: {selected_node}")
        plt.axis('off')  # Turn off the axis
        
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()  # Remove the previous canvas

        self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        logging.info("Graph plotted successfully.")

    def save_graph(self):
        selected_node = self.dropdown.get()
        if selected_node:
            plt.figure(figsize=(12, 12))
            self.plot_graph(selected_node)
            plt.savefig(f"{selected_node}_connections.png")
            messagebox.showinfo("Success", f"Graph saved as {selected_node}_connections.png")
        else:
            messagebox.showwarning("Warning", "Please select a neuron or token first.")

def main():
    logging.info("Loading model and tokenizer data...")
    
    try:
        with open('trained_model.json', 'r') as model_file:
            model_data = json.load(model_file)
        logging.info("Model data loaded successfully.")
        
        with open('trained_tokenizer.json', 'r') as tokenizer_file:
            tokenizer_data = json.load(tokenizer_file)
        logging.info("Tokenizer data loaded successfully.")

        root = tk.Tk()
        app = NeuralNetworkGraphApp(root, model_data, tokenizer_data)
        root.mainloop()

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
