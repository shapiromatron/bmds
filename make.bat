@ECHO off

if "%~1" == "" goto :help
if /I %1 == help goto :help
if /I %1 == lint goto :lint
if /I %1 == format goto :format
if /I %1 == test goto :test
if /I %1 == test-refresh goto :test-refresh
goto :help

:help
echo.Please use `make ^<target^>` where ^<target^> is one of
echo.  test         run python tests
echo.  test-refresh removes mock requests and runs python tests
echo.  lint         perform both lint-py and lint-js
echo.  format       perform both format-py and lint-js
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

:test-refresh
rmdir /s /q .\tests\cassettes
py.test
goto :eof
