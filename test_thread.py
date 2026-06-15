import sys, os, threading, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

from ui.globals import _write_console as _log, start_capturing_prints, custom_print, _original_print
import builtins

# Test 1: _log directly from main thread
_log('[TEST1] _log from main thread')

# Test 2: _log from a thread
def thread_func():
    _log('[TEST2] _log from thread')
    print('[TEST3] print from thread (before start_capturing)')
t = threading.Thread(target=thread_func)
t.start()
t.join()

# Test 3: After start_capturing_prints
start_capturing_prints()
print('[TEST4] print from main thread AFTER capturing')
t2 = threading.Thread(target=lambda: print('[TEST5] print from thread AFTER capturing'))
t2.start()
t2.join()

# Test 4: verify print is really custom_print
print('print is custom_print:', builtins.print is custom_print)
print('print is _original_print:', builtins.print is _original_print)

print('[TEST6] All tests done')
