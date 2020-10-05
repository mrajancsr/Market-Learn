from collections import MutableMapping

class MapBase(MutableMapping):
	"""Abstract base class that includes nonpublic Item Class"""

	class Item:
		__slots__ = "_key", "_value"

		def __init__(self, k,v):
			self._key = k
			self._value = v 

		def __eq__(self, other):
			return self._key == other._key

		def __ne__(self, other):
			return not (self == other)

		def __lt__(self, other):
			return self._key < other._key 

class UnsortedTableMap(MapBase):
	"""map implementation using an unordered list"""
	def __init__(self):
		self._table = []

	def __getitem__(self, k):
		"""return value associated with key k (raise keyError if not found)
			if found, takes O(1) time.  Otherwise:
        	O(n) time
        """
		for item in self._table:
			if item._key == k:
				return item._value 
		raise KeyError("key Error: " + repr(k))

	def __setitem__(self, k, v):
		for item in self._table:
			if item._key == k:
				item._value = v 
				return 
		self._table.append(self.Item(k, v))

	def __delitem__(self, k, v):
		for idx, item in enumerate(self._table):
			if item._key == k:
				self._table.pop(idx)
				return 
		raise KeyError("key Error: " + repr(k))

	def __len__(self):
		return len(self._table)

	def __iter__(self):
		for item in self._table:
			yield item._key
	def keys(self):
		return tuple(i for i in self)

class HashMapBase(MapBase):
	"""Abstract base class for map using hash-table with MAD compression"""
	pass
def TestUnsortedTableMap():
	chk = UnsortedTableMap()
	chk['hello'] = 'there'
	chk[3] = 5
	print(chk.keys())

TestUnsortedTableMap()