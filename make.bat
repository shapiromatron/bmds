@ECHO off

if "%~1" == "" goto :help
if /I %1 == help goto :help
if /I %1 == lint goto :lint
if /I %1 == format goto :format
if /I %1 == test goto :test
if /I %1 == test-refresh goto :test-refresh
if /I %1 == dist goto :dist
goto :help

:help
echo.Please use `make ^<target^>` where ^<target^> is one of
echo.  test         run python tests
echo.  lint         perform both lint-py and lint-js
echo.  format       perform both format-py and lint-js
echo.  dist         builds source and wheel package
goto :eof

:lint
black . --check && flake8 .
goto :eof

:format
black . && isort -q . && flake8 .
goto :eof

:test
py.test
goto :eof

:dist
rmdir /s /q .\bmds.egg-info
rmdir /s /q .\dist
python setup.py clean --all
python setup.py bdist_wheel
dir .\dist
goto :eof
