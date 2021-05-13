all: env build host

env:
	source env/bin/activate
build:
	buildozer android debug
host:
	buildozer serve
