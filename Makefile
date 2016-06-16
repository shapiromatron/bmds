.PHONY: init

init:
	pip install -r requirements.txt

test:
	# This runs all of the tests.

	# To run an individual test, use the -k flag to grep for matching:
	# 	$ py.test -k test_monkey_has_tail
	#
	# To show print statements when debugging tests, use the -s flag:
	# 	$ py.test -k test_monkey_has_tail -s
	#
	py.test tests
