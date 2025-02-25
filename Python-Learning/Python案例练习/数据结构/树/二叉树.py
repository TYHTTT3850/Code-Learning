from collections import deque #collections 是 Python 的一个标准库，提供了多种额外的容器数据类型

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

"""--------------------层序遍历(广度优先遍历，从上到下，从左到右)--------------------"""
# 借助队列实现
def level_order(root):
    if not root:
        return []

    result = [] #初始化结果列表为空列表
    queue = deque()
    queue.append(root) #根节点入队

    while queue:
        level = [] #用于存储当前层的节点
        for i in range(len(queue)): #遍历队列中所有的节点
            node = queue.popleft() #队首节点出队
            level.append(node.val)
            if node.left: queue.append(node.left) #左子节点入队
            if node.right: queue.append(node.right) #右子节点入队
        result.append(level) #将当前层的节点值存入结果中
    return result

"""--------------------前序遍历(深度优先遍历)--------------------"""
# 借助递归实现
def pre_order(root):
    result = []
    if not root:
        return
    #根节点->左子树->右子树
    result.append(root.val)
    result.append(pre_order(root.left))
    result.append(pre_order(root.right))
    return result

"""--------------------中序遍历(深度优先遍历)--------------------"""

def in_order(root):
    result = []
    if not root:
        return
    #左子树->根节点->右子树
    result.append(in_order(root.left))
    result.append(root.val)
    result.append(in_order(root.right))
    return result

"""--------------------后序遍历(深度优先遍历)--------------------"""

def post_order(root):
    result = []
    if not root:
        return
    # 左子树->右子树->根节点
    result.append(post_order(root.left))
    result.append(post_order(root.right))
    result.append(root.val)
    return result

"""--------------------初始化二叉树--------------------"""
# 初始化节点
n1 = TreeNode(1)
n2 = TreeNode(2)
n3 = TreeNode(3)
n4 = TreeNode(4)
n5 = TreeNode(5)

# 构建节点之间的连接
n1.left = n2
n1.right = n3
n2.left = n4
n2.right = n5

"""--------------------插入与删除节点--------------------"""
p = TreeNode(0)
# 在n1和n2之间插入p节点
n1.left = p
p.left = n2
# 删除p节点
n1.left = n2

"""--------------------具体调用--------------------"""
#层序遍历
result1 = level_order(n1)
print(result1)

#前序遍历
result2 = pre_order(n1)
print(result2)

#中序遍历
result3 = in_order(n1)
print(result3)

#后序遍历
result4 = post_order(n1)
print(result4)