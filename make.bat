@ECHO off

if "%~1" == "" goto :help
if /I %1 == help goto :help
if /I %1 == lint goto :lint
if /I %1 == format goto :format
if /I %1 == test goto :test
if /I %1 == dist goto :dist
goto :help

:help
echo.Please use `make ^<target^>` where ^<target^> is one of
echo.  test         run python tests
echo.  lint         check formatting issues
echo.  format       fix formatting issues where possible
echo.  dist         builds source and wheel package
goto :eof

:lint
ruff format . --check && ruff .
goto :eof

:format
ruff format . && ruff . --fix --show-fixes
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
