from linkedlist import SinglyLinkedList, DoublyLinkedList

def main():
    l1 = SinglyLinkedList()
    l2 = SinglyLinkedList()

    l1.insert_at_start(1)
    l1.insert_at_start(2)
    l1.insert_at_start(3)
    l2.insert_at_start(3)
    l2.insert_at_start(5)
    l2.insert_at_start(9)
    print(l1.start_node.element)

    
    


def main2():
    dl = DoublyLinkedList()
    dl.insert_at_end(22)
    dl.insert_at_end(56)
    dl.insert_at_end(190)
    dl.insert_at_start(200)
    dl.insert_after_value(56,'raj')
    dl.insert_after_value(190,2000)
    dl.traverse()
    dl.delete_by_value(2000)
    print("after deleting")
    dl.traverse()
    dl.reverse()
    print('after reversing')
    dl.traverse()


s = [100,1,2,2,2,10,10,10,22,22,10,1]



def main3():
    from linked_collections import LinkedStack,LinkedQueue,LinkedDeque

    ls = LinkedStack()
    ls.push("raju")
    ls.push("prema")
    ls.push(2)
    ls.traverse()
    print("after deleting element")
    ls.pop()
    ls.traverse()

    # testing linkedDeque
    print("testing linkedQueue")
    lq = LinkedQueue()
    lq.enqueue("raju")
    lq.enqueue("prema")
    lq.enqueue("what the fuck")
    print("after traversing")
    lq.traverse()
    print("after removing the element")
    lq.dequeue()
    lq.traverse()
    print("linkedqueue adding after deletion")
    lq.enqueue("fuck prema")
    lq.traverse()
    print("testing linkedDeque")
    ld = LinkedDeque()
    ld.add_front("raju")
    ld.add_front("prema")
    ld.add_front("prema is stupid")
    ld.add_rear("raj is tha bomb")
    print("\n")
    print("lets traverse")
    ld.traverse()
    print("\n")
    print("testing deletion")
    ld.remove_front()
    print("after removing front")
    #ld.remove_rear()
    ld.traverse()
    print("adding items")
    ld.add_front(3)
    ld.add_rear(100)
    print("\n")
    ld.traverse()
    print("after removing rear")
    ld.remove_rear()
    print("\n")
    ld.traverse()
    print("\n")
    print("testing addition after removal of rear")
    ld.add_front(55)
    ld.add_rear(66)
    ld.traverse()
    print("success test")
main()
"""
from linked_collections import PositionalList
pl = PositionalList()
pl.add_first(3)
pl.add_last(100)
for i in pl:
    print(i)

"""