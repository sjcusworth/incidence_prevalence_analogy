rm -r build; rm -r dist
pyinstaller run.py --noconfirm -n incprev_analogy --add-data="./config.yml:."
mv dist/incprev_analogy ./
zip -r incprev_analogy_linux.zip incprev_analogy
rm -r build; rm -r dist; rm -r incprev_analogy; rm incprev_analogy.spec
