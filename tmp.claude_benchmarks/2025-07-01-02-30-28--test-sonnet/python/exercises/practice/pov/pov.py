from json import dumps


class Tree:
    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children is not None else []

    def __dict__(self):
        return {self.label: [c.__dict__() for c in sorted(self.children)]}

    def __str__(self, indent=None):
        return dumps(self.__dict__(), indent=indent)

    def __lt__(self, other):
        return self.label < other.label

    def __eq__(self, other):
        return self.__dict__() == other.__dict__()

    def from_pov(self, from_node):
        # Check if node exists
        if not self._node_exists(from_node):
            raise ValueError("Tree could not be reoriented")
        
        # Build adjacency list representation
        adjacency = {}
        self._build_adjacency(adjacency)
        
        # Build new tree using DFS from the target node
        visited = set()
        return self._build_tree_from_node(from_node, adjacency, visited)

    def path_to(self, from_node, to_node):
        # First reparent from from_node
        try:
            reparented = self.from_pov(from_node)
        except ValueError as e:
            # If the error is from from_pov, preserve the original message
            raise e
        
        # Then find path from root to to_node
        path = reparented._find_path_to_node(to_node)
        if path is None:
            raise ValueError("No path found")
        
        return path
    
    def _node_exists(self, target):
        """Check if a node with the given label exists in the tree"""
        if self.label == target:
            return True
        return any(child._node_exists(target) for child in self.children)

    def _build_adjacency(self, adjacency):
        """Build bidirectional adjacency list"""
        if self.label not in adjacency:
            adjacency[self.label] = []
        
        for child in self.children:
            adjacency[self.label].append(child.label)
            if child.label not in adjacency:
                adjacency[child.label] = []
            adjacency[child.label].append(self.label)
            child._build_adjacency(adjacency)

    def _build_tree_from_node(self, node_label, adjacency, visited):
        """Build tree rooted at node_label using DFS"""
        visited.add(node_label)
        
        children = []
        for neighbor in adjacency.get(node_label, []):
            if neighbor not in visited:
                child_tree = self._build_tree_from_node(neighbor, adjacency, visited)
                children.append(child_tree)
        
        return Tree(node_label, children)

    def _find_path_to_node(self, target):
        """Find path from root to target node"""
        if self.label == target:
            return [self.label]
        
        for child in self.children:
            path = child._find_path_to_node(target)
            if path:
                return [self.label] + path
        
        return None
