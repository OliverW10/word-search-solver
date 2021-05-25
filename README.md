# word-search-solver

This is a App for android meant to solve a word search (find-a-word) puzzle from a photo.
It is made with Python using [kivyMD](https://github.com/kivymd/KivyMD) for the GUI, [OpenCV](https://docs.opencv.org/master/d0/de3/tutorial_py_intro.html) for the vision and builds to android with [buildozer](https://github.com/kivy/buildozer)

The vision isolates letters using [contours](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_begin/py_contours_begin.html#contours-getting-started) and classifies them using [k-Nearest Neighbour](https://docs.opencv.org/master/d0/d72/tutorial_py_knn_index.html) so that it can place them in a grid and search for words.

You can copy the APK file in /bin to your android phone and run it to install the app

TO DO:
 - Make Solver Based on position not grid

When cloning use `git clone --depth 1 https://github.com/OliverW10/word-search-solver.git` to reduce size
to get the test images use `git submodule init` and then `git submodule update`
