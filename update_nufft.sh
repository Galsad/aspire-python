rm nufft$1df90.pyf
echo "done removing!"

f2py nufft$1df90_f2py.f -m nufft$1df90 -h nufft$1df90.pyf
echo "pyf file was created"

f2py -c nufft$1df90.pyf nufft$1df90_f2py.f dfftpack.f next235.f
echo "Done"