# file: $NEDC_NFC/class/python/nedc_qml_tools/Makefile
#

# define source and object files
#
SRC = $(wildcard *.py)

# define an installation target
#
install:
	cd $(NEDC_NFC)/lib; rm -f $(SRC)
	cp -f $(SRC) $(NEDC_NFC)/lib/
	cd $(NEDC_NFC)/lib; chmod u+rw,g+rw,o+r $(SRC)

#
# end of file
