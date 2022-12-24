# OS
## Basics
- Unix: Time-sharing operating system, not free.
- GNU/Linux: free. GNU stands for GNU is not Unix.
    - Kernel space: interact with the hardware
    - User space: contains applications
    - Operating system: connects user space and kernel space
- Shell: The application where you enter commands
    - Bourne Shell: `bash`, `zsh`, `ksh`
    - C shell: `csh`, `tcsh`
    - To remotely access a shell session you use `ssh` (secure shell)
- Editors:
    - vim: the to-go editor
    - nano: minimal learning curve

## Shell 
### Commands
- File system
    - `.`: current dir
    - `..`: parent dir
    - `~`: home dir
    - `/`: root dir
    - `-`: dir I was just in
    - `pwd`: show path
    - `mkdir`: make dir
    - `ls -l`: long format with more info
    - `ls -a`: show hidden files
    - `ls -F`: show directories with a `/` and excecutables with a `*`
    - `rm -r`: recursive
    - `cp -r dir1 dir2`: copy
    - `mv file1 file2`: rename
    - `touch file`: create
    - `cat file`: output the contents
    - `wc`: count, `-l` for lines, `-w` for words, `-c` for characters
- System
    - `shutdown`: shutdown
    - `reboot`: restart
    - `date`: show current date and time
    - `whoami`: who you are logged in as
    - `finger user`: display info about user
    - `man *command*`: manual for command
    - `whatis *command*`: more info about a command
    - `df`: disk usage
    - `du`: directory space usage
- Compression
    - `tar cf file.tar files`: create a tar named file.tar containing files
    - `tar xf file.tar`: extract files from file.tar
- Searching
    - `grep pattern files`: search for pattern in files
    - `grep -r pattern dir`: search recursively for pattern in dir
    - `find *file*`: find files
- Permission
    - `chmod *octal* *file*`: change permission
        - `777`: read, write and excetute for all
- others
    - `alias name 'command'`: aliasing
    - `|`: pipes

### Other stuff
- Customization
    - Interactive login shell:
        - `/etc/profile`
        - `~/.bash_profile`
        - `~/.bash_login`
        - `~/.profile`
    - Non-interactive shell:
        - `~/.bashrc`
    - Conclusion:
        - Usual user: edit `~/.bashrc`
        - `zsh` user: edit `~/.zshrc`
- Redirection of I/O streams
    - `>`: redirect `stdout` to a file, overwrite
    - `>>`: redirect `stdout` to a file, append
    - `<`: redirect file to `stdin`
    - `dev/null`: data sink where data is discarded
- Jobs/processes management
- Environment variables: 
    - `export`: save new-defined variables to global
    - `unset`: delete the variable
    - `PATH`: The list of locations that shell search for a command, separated by `:`
    `PATH=$HOME/bin:$HOME/.local/bin:$PATH`\
    `export PATH`
    This adds the following paths to PATH:
        - `$HOME/bin`: the default location for executables
        - `$HOME/.local/bin`: the default location for python to install packages

### Scripting
[Bash scripting cheatsheet](https://devhints.io/bash)\
[Bash guide for beginners book](https://tldp.org/LDP/Bash-Beginners-Guide/html/index.html)
- Special variables:
    - `$PWD`: path to here
    - `$0`: file name
    - `$@`: expands to quoted arguments
    - `$1-9`: the first nine arguments
    - `$#`: the number of arguments
- Strings
    - Single-quoted strings: hard-coded
    - Double-quoted strings: soft-coded (can have references)
- First line: specify interpreter, starts with a shebang\
    `#!interpreter_command [optional arguments]`
    Example: `#!/usr/bin/env bash`
- For loop: 
    - Command substitution: `$(...)` excecutes the codes inside in a new shell and write to stdout
- If conditionals: write conditionals in `[]`
    - String comparison
        - `==`: equal
        - `!=`: not equal
        - `-z`: True if length is zero
        - `-n`: True if length is not zero
    - Integer comparison
        - `-eq`: = 
        - `-ne`: !=
        - `-lt`: <
        - `-le`: <=
        - `-gt`: >
        - `-ge`: >=
    - Files condition: `[*command* *file*]`
        - `-d`: True if FILE exists and is a directory
        - `-f`: True if FILE exists and is a regular le
        - `-e`: True if FILE exists
        - `-r`: True if FILE exists and is readable
        - `-w`: True if FILE exists and is writable
        - `-x`: True if FILE exists and is executable

## git
[git command reference](https://git-scm.com/docs)\
[pro fit book](https://git-scm.com/book/en/v2)
- Before you start
        [user]
        name = FirstName LastName
        email = you@domain.com
        [core]
        editor = vim 
- Basics
    - `git init`: initialize a git repo
    - `git clone *URL*`: download
    - `git status`: status cummary
    - `git log`: see the info for commits. `q` to quit.
- branching: you should never directly commit to master branch
    - `git branch *branch_name*`: create new branch
    - `git checkout *branch_name*`: go to the branch
    - `git checkout -b *branch_name*`: combination of the previous commands
    - `git diff *branch_name* *new_branch_name*`
    - `git merge *new_branch_name*`: merge
    - `git branch -d *branch_name*`: delete branch
- Updating
    - `git add *file_name*`: stash file
    - `git rm --cashed *file_name*`: remove file from `.git`. Remember to commit.
    - `git rm *file_name*`: remove from both `.git` and local file system.
    - `git commit -m *your_message*`: commit
    - `git commit`: commit with `vim`
        - `i`: enter insert mode
        - Enter your message
        - press `esc` to exit insert mode
        - `:wq`: enter command line mode, write and quit
    - `git push --set-upstream origin *branch_name*`: push to remote repo
    - `git tag 1.0.0 *first_10_digits_of_hash_id*`: tag
- Reverting
    - `git revert *hash_id*`: revert
    - `git reset *file_name*`: reset the changes you commited but haven't pushed
- Remote
    - `git remote -v`: check status of the remote
    - `git remote add *remote_name* *remote_URL*`: add new remote site
    - `git pull *remote_name* *branch_name*`: pull
- Housecleaning
    - `.gitignore`: add unnecessary files like `.DS_Store` so git does not track it.
        - `*.pdf`: often useful to ignore bynary files
        - `git add --force important.pdf`: force add
        - `!*.py`: don't ignore any python files

## Regex
- basic commands
    - `.`: any one character except a newline
    - `*`: zero or more occurrence of the preceeding character
    - `+`: one or more occurrence of the preceeding character
    - `?`: exactly one occurrence of the preceeding character
    - `-`: escape for special character
    - `()`: capture a group (match all)
    - `|`: logical OR
    - `{}`: numerical range for the number of occurrences
    - `[]`: character group (match any)
- Convenience classes
    - `\d`: digit
    - `\D`: non-digit
    - `\w`: a word, including letter and digit
    - `\W`: non-word
    - `\s`: white-space
    - `\S`: non-whitespace
    - `^`: beggining of line
    - `$`: end of line
    - `\b`: word boundary
    - `\B`: non-word boundary

## vim
- Modes
    - Normal mode
    - Insert mode: mode where you can insert texts. Press `i` to enter and `esc` to quit.
    - Command-line mode: Press `:` to enter from normal mode
- Useful commands
    - `:q!`: quit without saving
    - `:wq`: save and quit
    - `:wqa`: save all open files and quit
    - `/*pattern*`: search for pattern (can be regex). `n` for the next word match and `N` for the previous word match
    - `dd`: delete the line where the cursor is on
    - `yy`: copy the line where the editor is on
    - `I,i,a,A`: insert text: at beginning of line (`I`), before the cursor (`i`), after the cursor (`a`), at the end of line (`A`)
    - `p`: paste the last text
    - `gg`: go to first line
    - `G`: go to last line

# Python
## Basics
[python tutor](https://pythontutor.com/) for visualizing python codes
[yet another python formatter](https://github.com/google/yapf) for formatting
- Typing
    - python is strongly typed: rare implicit conversion of type
    - python is dynamically typed: type-checking is performed at runtime not complie time.
- Functions and Environments
    - Rebinding a mutable object (list a list) creates a new object
    - Rebinding an immutable object maintains the reference


## OOP
- Encapsulation
    - `_*var`: common way to create (pseudo) private variables (no private variables exist in python)
- Polymorphism
    - `super().*method*()`: way to call the method in super-class from sub-class
- dunder (under under methodname under under) methods: writing new classes with these method defined lead to great functionality:
    - strings:
        - `__repr__`: for debugging
        - `__str__`: for informing
    - arithmetics:
        - `__add__`: +
        - `__eq__`: =
    - helpers:
        - `__dir__`: a list of valid attributes
    - sequencing:
        - `__len__`: length
        - `__getitem__`: get item
        - `__setitem__`: set item
- Variable binding and decorator
    - `import *` binds all names except `_*name*`, which may only be used at the module level
    - In nested environments, inner functions have access to the variables in the scope of outer functions. Example:
    ```python
    def set_partial_value(partial):
        def set_final_value(final):
            return ' '.join((partial, final))
                return set_final_value # we return a function here
    i_am = set_partial_value('Hi, my name is')
    print(i_am('Alice'))
    ```
    Another example: wrap a timer around function f to time the function
    ```python
    def timer(f):
        def inner(*args, **kwargs):
            t0 = time.time()
            retval = f(*args, **kwargs) 
            elapsed = time.time() - t0
            print(f"{f.__name__}: elapsed time {elapsed:e} seconds")
        return retval
    return inner
    def f(x):
        return x
    timed_f = timer(f)
    ```
    - decorator = outer function + closure (that wraps code around the captured function). In the example above:
        - outer function: `timer`
        - closure: `inner`
        - code: `t0=time.time()`, etc
        - captured function: `f`
    - New pythonic syntax:(so we don't need a new name)
    ```python
    def timer(f):
        def inner(*args, **kwargs):
            t0 = time.time()
            retval = f(*args, **kwargs) 
            elapsed = time.time() - t0
            print(f"{f.__name__}: elapsed time {elapsed:e} seconds")
        return retval
    return inner
    @timer
    def f(x):
        return x
    ```
- special methods
    - `@classmethod`: for methods that do not depend on the state of an object. Example:
    ```python
    def from_polar(cls, r, phi):
        return cls(r * np.cos(phi), r * np.sin(phi))
    ```
    - `@staticmethod`: for methods that do not depend on class type.
    - class variable: no `self.` at initialization, global to all instances of this class, can be overloaded

## Advanced data structures
- Iterators
    - Structured like a linked list
    - Iterators themselves implement the `__iter__` method (i.e. the `__itet__` method should return the iterator)
    - Iterators implement the `__next__` method 
- Generators
    - Works like iterators, but are "lazy"
    - Calling `g.next()` return the next yield. Can also be iterated via a for loop.
    ```python
    class LinkedList:
        # other code skipped
        def __iter__(self):
            node = self.first
            while node != None:
                yield node
            node = node.next
    ```
    - Generator funtion example:
    ```python
    def g():
        n=0
        while n < 10:
            yield n
            n += 1
    ```
    - Generator expression example:
    ```python
    ge = (x for x in g())
    ```
    - Useful built-in generators, most of them available in `itertools`:
        - Filters
        - Maps
        - Merge of inputs
        - Expansion of input into multiple outputs
        - Rearrangements
        - Reductions
- Coroutines: the more general version of functions where two coroutines are symmetric: they can reprtitively call each other with different input each time.
```python
from inspect import getgeneratorstate
def coroutine():
    ncalls = 0
    while True:
        x = yield
        ncalls += 1
        print(f'coroutine(): {x} (call id: {ncalls})')
def main():
    c = coroutine()
    print(getgeneratorstate(c))
    next(c) # prime the coroutine
    print(getgeneratorstate(c))

    c.send('Hello')
    print('main(): control back in main function')
    last_call = c.send('CS107')
    print(f'main(): called coroutine() {last_call} times'

    c.close()
    print(getgeneratorstate(c))

if __name__ == "__main__":
    main()
```
output:
```
GEN_CREATED
GEN_SUSPENDED
coroutine(): Hello (call id: 1)
main(): control back in main function
coroutine(): CS107 (call id: 2)
main(): called coroutine() 2 times
GEN_CLOSED
```

## Modules and Packages
- modules: 
    - your own modules like `*module*.py`
    - python standard modules
    - third-party modules
- packages: hierarchy of modules
    - `if __name__ == "__main__":`: only runs when this module is passed to the interpreter.
    - `__init__.py`: things you should do when you first enter this directory.
- Python Package Index (PyPI): the remote server to fetch the software from.
    - `python -m pip install *package*`
- Install and distribute python packages

## python internals
- Objects: every piece of data is stored in an object. Functions are first-class objects
    - identity: its location in memory
    - type: the internal representation as well as methods and operations it supports
    - value: mutable or immutable
- Interpretors: turn code objects into frame objects and execute them
- Dynamic memories: python prioritizes flexibility and the possibility for fast prototyping. Spatial and temporal locality of the data is not optimal.

# Databases
## Basics
- Types of databases:
    - Relational: `SQL` and its derivatives
        - The central element is a `table`
        - Multiple tables relate to each other via common values in columns, called `keys`
    - Document orianted: `CoudhDB`, `MongoDB`
        - Stores the data in nested models
        - Examples: `XML`, `YAML`, `JSON`
    - Key-value: `Riak`, `Memcached`, `leveldb`
        - Uses a dictionary like data structure
        - More flexible and follows more closely modern concepts like OOP
    - Graph oriented: `Meo4J`
    - Columnar: `HBase`

## SQL (Structured Query Language)
- Concepts
    - `primary key`: a column or combination of columns in a table whose values uniquely identify each row of the table. A table has only one primary key.
    - `foreign key`: a column of combination of columns in a table whose values are a primary key value for some other table.
    - A primary/foreign key combination creates a parent/child relationship between the tables that contain them
- Core Commands
    - `SELECT FROM`: select a table name
    - `INSERT INTO`: Insert data into the table
    - `UPDATE`: Change data values in the table
        - `SET`:
    - `DELETE`: delete data in the table
- Structural Commands
    - `CREATE`: create a table in the database
    - `DROP`: delete a table in the database
    - `ALTER`: add, delete or modify columns in an existing table
        - `ADD`
- Conditionals:
    - `IF`:
    - `WHERE`: filter values
    - `ORDER BY`:
    - `DISTINCT`:
    - `LIMIT`:
    - `OFFSET`:
- Aggregates:
    - `COUNT`:
    - `AVG`:
    - `SUM`: 
- Merges:
    - `INNER JOIN`: column value in both tables
    - `LEFT/RIGHT JOIN`: column value in left/right table
    - `OUTER JOIN`: column value in either table

# Other MLE things
## Software licenses
[guide](https://choosealicense.com/)
- Copyleft: copytighted with additional distribution terms that anyone who use the code must have the distribution terms unchanged.
    - GUN General Public License v3.0 (GNU GPLv3)
        - bash shell
    - MIT license
        - zsh shell

## Virtual Environments
- Conda
    - `conda env list`: list the environments
    - `conda create --name *env_name* python=3.7`: create an environment
    - `conda activate *env_name*`: activate environment
    - `conda install *package_name*=*version*`: install packages
    - `conda deactivate`: deactivate
    - `conda env export > *my_env*.yml`: export environment settings
    - `conda env create -f *my_env*.yml`: create environment from yml file
-   Docker
    - `docker --version`: check version
    - `docker container/image ls`: list containers/images
    - `docker build -t *image_name* -f *file_name* .`: build image ffrom Dockerfile at location `.`
    - `docker run --rm --name *container_name* -ti --entrypoint /bin/bash *image_name*`: run container from image. 
        - `--rm`: clean up the container and remove the file system when the container exit.
        - `-ti`: give us a terminal with interactive mode
        - `--entrypoint`: default command to execute on startup
    - `docker system prune -a`: remove all images not referenced by a container

## Continuous Integration

## Performance
- Debugging
    - Common debugging techniques
        - Interactive debugging: requires a debugger like `gdb`
        - Trace based debugging: use `print`
        - Online debugging: attach to a running process to isolate  problem
        - Isolation by bisection (e.g. `git bisect`)
        - Inspection after a failure
    - `pdb`: the python debugger.
- Profiling
    - `cProfile`: python profiling package

## Testing
- Test-Driven Development (TDD): first develop Software Requirements Specification (SRS), then write tests, and finally write implementations.
- Types of tests
    - Unit tests: smallest tests applied to classes and functions in a module and sometimes the module itself.
    - Integration tests: combine unit tests that have a dependency on each other
    - Regression tests: re-run unit tests to make sure the integration did not break any core funcionalities
    - System and acceptance tests: larger tests that take place upon multi-mode completion.
- Conventions
    - Write your tests in a different dir than your `src`
    - Test file should be named like `test_*filename*`
    - Write a shell script to run all tests
- Packages
    - `unittest`: 
    ```python
    """
    This test suite (a module) runs tests for subpkg_1.module_1 of the cs107_package.
    """
    import unittest # python standard library
    # project code (import into this namespace)
    from cs107_package.subpkg_1.module_1 import *
    class TestTypes(unittest.TestCase):
        def test_class_Foo(self):
            """
            This is just a trivial test to check that `Foo` is initialized
            correctly. More tests associated to the class `Foo` could be written in
            this method.
            """
            f = Foo(1, 2)
            self.assertEqual(f.a, 1)
            self.assertEqual(f.b, 2)

    class TestFunctions(unittest.TestCase):
        def test_function_foo(self):
            """
            This is just a trivial test to check the return value of function `foo`.
            """
            self.assertEqual(foo(), "cs107_package.subpkg_1.module_1.foo()")

    if __name__ == '__main__':
    unittest.main()
    ```
    - `pytest`:
    ```python
    """
    This test suite (a module) runs tests for subpkg_1.module_2 of the cs107_package.
    """
    import pytest # these tests are designed for pytest
    # project code
    from cs107_package.subpkg_1.module_2 import *
    class TestFunctions:
        """We do not inherit from unittest.TestCase for pytest's!"""
        def test_bar(self):
            """
            This is just a trivial test to check the return value of function `bar`.
            """
            assert bar() == "cs107_package.subpkg_1.module_2.bar()"

    def example_function():
        """If you have code that raises exceptions, pytest can verify them.
        """
        raise RuntimeError("This function should not be called")

    def test_example_function():
        with pytest.raises(RuntimeError):
        example_function()
    ```
- Code coverage: report how much of your source code have your test covered.

## Documentation

# Deep Learning Architectures
## Custom Classes
- Modules
```python
class MLP(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class nn.Module to perform
        # the necessary initialization
        super().__init__()
        self.net = nn.Sequential()

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.net(X)

# Modules can be nested:
super_module = nn.Sequential(nn.LazyLinear(),MLP(),nn.Dropout())
```
- Parameters
```python
# Access parameters
[(name, param.shape) for name, param in net.named_parameters()]

# Parameter initialization
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```
- Layers
```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```
- Saving and Loading
```python
# Saving
torch.save(net.state_dict(), 'mlp.params')
# Loading
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

## Devices
- View GPU information: first install [Nvidia driver and CUDA](https://developer.nvidia.com/cuda-downloads) then run
```python
!nvidia-smi
```
- Set devices
```python
def cpu():
    return torch.device('cpu')
def gpu(i=0):
    return torch.device(f'cuda:{i}')
def num_gpus():  
    return torch.cuda.device_count()
def try_gpu(i=0):
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()
def try_all_gpus():
    return [gpu(i) for i in range(num_gpus())]
```
- Storing on gpu
```python
Y = torch.rand(2, 3, device=try_gpu(1))
net = net.to(device=try_gpu())
```

## Optimizing Structures
- Hybridizing imperative and sympolic programming:
```python
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net
x = torch.randn(size=(1, 512))
net = get_net()
net = torch.jit.script(net)   # Very easy to optimize!
net(x)
```

- Asynchronous computation
- Automatic Parallelism
- Training on multiple GPUs
- Parameter servers