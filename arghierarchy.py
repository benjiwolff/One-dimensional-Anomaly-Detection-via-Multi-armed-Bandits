from board import Board


class ArgHierarchy:
    def __init__(self, algorithm:str) -> None:
        self.__dict__['dictionary'] = {}
        self.algorithm = algorithm
        self.dist_parameters = {}
    
    def __setattr__(self,key,value) -> None:
        if key in Board.dist_parameters:
            self.dist_parameters[key] = value
        else:
            self.dictionary[key] = value
        super().__setattr__(key,value)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)