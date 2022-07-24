class Node : 
    """
    Class rappresenting a node in a decision Tree
    """
    def __init__(self, feature = None, thresh = None,  left = None, right = None, *, value = None) : 
        """
        Args: 
            {int} Feature    : which feature split is performed on
            {float32} thresh : threshold for the split 
            {Node} left      : left child 
            {Node} right     : right child
            {int} value      : If is leaf this is the predicted label
        Return:
            {Node} : Initialize the Node Object
        """

        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.thresh = thresh
    
    def is_leaf(self): 
        """
        Return: 
            {bool} : True if this node is a leaf node
        """
        return self.value is not None