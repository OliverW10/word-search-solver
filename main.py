# import logging
# logging.getLogger("kivy").setLevel(logging.ERROR)
# logging.getLogger("keras").setLevel(logging.ERROR)
# import os
# import sys
# sys.stderr = open('output.txt', 'w')
# sys.stdout = sys.stderr
# 
# from kivy.logger import LoggerHistory
# print('\n'.join([str(l) for l in LoggerHistory.history]))
# os.environ["KIVY_NO_CONSOLELOG"] = "1"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.clock import Clock
from kivy.graphics import *
from kivy.core.window import Window
from kivy.graphics.transformation import Matrix
from kivy.lang.builder import Builder
from kivy.properties import *
from kivy.graphics.texture import Texture
from kivy.utils import platform
import time
import numpy as np
import image_to_numpy
# import cv2

from solvers import Solvers
from imageReader import ImageProcessing

# kivy.require("2.0")
if platform == "android":
	from android.permissions import request_permissions, Permission
	from android.storage import primary_external_storage_path
	request_permissions([Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE])

### step 1 ###
class StartPage(GridLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 1

		self.title = Label(text = "Find-a-word Solver")
		self.title.size_hint = (1, 0.2)
		self.add_widget(self.title)

		self.buttons = GridLayout()
		self.buttons.cols = 2

		self.loadButton = Button(text="Load File")
		self.loadButton.bind(on_press = self.loadImage)
		self.buttons.add_widget(self.loadButton)

		self.cameraButton = Button(text = "Camera (WIP)")
		self.cameraButton.bind(on_press = self.launchCamera)
		self.buttons.add_widget(self.cameraButton)

		self.add_widget(self.buttons)

	def loadImage(self, instance):
		self.title.text = "test"
		print("go to load image page")
		self.caller.screen_manager.current = "Load"

	def launchCamera(self, instance):
		print("go to camera page")
		self.caller.screen_manager.current = "Camera"

### step 2, load###
class LoadPage(GridLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 1
		self.file_thing = FileChooserIconView()
		self.file_thing.bind(on_submit=self.choseFile)
		if platform == "android":
			# self.file_thing.path = primary_external_storage_path()
			self.file_thing.path = "data/data/org.test.myapp"
		else:
			self.file_thing.path =  "./tests/fulls"
		self.add_widget(self.file_thing)

	def choseFile(self, x, *args):
		self.caller.screen_manager.current = "LineUp"
		self.caller.line_up_screen.setImage(x.selection[0])
		# print("chosen file: ", x.path, x.selection)


class WordWidget(BoxLayout):
	def __init__(self, text, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 2
		self.textLabel = Label(text=text)
		self.add_widget(self.textLabel)

		self.removeButton = Button(text="x", size_hint_max_x=50)
		# self.removeButton.size_hint = (0.2, 0.2)
		self.removeButton.bind(on_press=self.remove)
		self.add_widget(self.removeButton)
		# self.size_hint_max_y = 0.02

	def remove(self, *args):
		self.caller.removeWord(self.textLabel.text)

### step 2, camera ###
class CameraPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)

		self.camera = Camera(play = False)
		self.add_widget(self.camera)
		self.camera.play = True
		self.camera.bind(on_texture=lambda x:print("new frame"))
		self.camera.bind(on_load=lambda x:print("camera started"))

		self.picButton = Button(text = "Take Picture", pos_hint = {"center_x":0.5, "y":0.125}, size_hint = (0.1, 0.1))
		self.picButton.bind(on_press = self.takePictue)
		self.add_widget(self.picButton)

		self.title = Label(text="Try to get the grid flat and square-on")
		self.title.size_hint = (1, 0.1)
		self.add_widget(self.title)

	def takePictue(self, name = "date"):
		img_name = time.strftime("%Y%m%d_%H%M%S")
		self.camera.export_to_png(f"./IMG_{img_name}.png")
		print("Captured "+f"./IMG_{img_name}.png")

	def cameraOn(self):
		self.camera.play = True

	def cameraOff(self):
		self.camera.play = False


def scaleNumber(n, x1, x2, y1, y2):
	range1 = x2-x1
	range2 = y2-y1
	ratio = (n-x1)/range1
	return ratio * range2+y1

### 3 ###
class LineUpPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.imgFilename = "temp_img.png"
		self.imgRect = [0, 0, 1, 1]

		self.movingLayout = Scatter()
		self.movingLayout.do_rotation = False
		self.add_widget(self.movingLayout)

		self.continueButton = Button(text="Continue", size_hint = (0.15, 0.1), pos_hint = {"x":0.85, "y":0.85})
		self.add_widget(self.continueButton)
		self.continueButton.bind(on_press = self.continued)
		self.on_touch_down = self.buttonTouchCheck

		self.squareMargin = 0.1
		# self.bind(on_size=lambda _:self.makeSquare(), on_pos=lambda _:self.makeSquare(self.squareMargin))
		Window.bind(on_resize=lambda *args:self.makeSquare(self.squareMargin))
		# self.movingLayout.bind(on_transform_with_touch=lambda *args:print(self.getPosCv()))
		self.createImgTexture()
		self.makeSquare(self.squareMargin)

	def buttonTouchCheck(self, touch, *args):
		print("Widget Pos", self.continueButton.pos, self.continueButton.size)
		print("Touch pos ", self.to_local(touch.pos[0], touch.pos[1]))
		print("Touch Check ", self.continueButton.collide_point(touch.pos[0], touch.pos[1]))
		if self.continueButton.collide_point(*touch.pos):
			self.continued()
		else:	
			return self.movingLayout.on_touch_down(touch)

	def setImage(self, img):
		print("path given: ",img)
		self.imgFilename = img
		self.createImgTexture()
		self.makeSquare(self.squareMargin)

	def checkSet(self):
		return imgFilename != "temp_img.png"

	def continued(self, *args):
		print(self.getPosCv())
		if self.checkSet:
			self.caller.words_screen.setStuff(self.imgFilename, self.getPosCv())
			self.caller.screen_manager.current = "Words"

	def getPosKivy(self):
		# gets the coordinates of each corner of the line-up square with the origin being bottom left in pixels
		size = min(self.imgSize[0]*Window.size[0], self.imgSize[1]*Window.size[1])*(0.5-self.squareMargin/2) # half of the side length of the line-up square
		midX, midY = Window.size[0]/2, Window.size[1]/2
		topLeft = self.movingLayout.to_parent(midX-size, midY+size)
		topRight = self.movingLayout.to_parent(midX+size, midY+size)
		bottomLeft = self.movingLayout.to_parent(midX-size, midY-size)
		bottomRight = self.movingLayout.to_parent(midX+size, midY-size)
		return topLeft, topRight, bottomRight, bottomLeft

	def getPosCv2(self):
		# gets the coordinates of each corner of the line-up square with the origin being top left as a percentage of the window size
		size = min(self.imgSize[0]*Window.size[0], self.imgSize[1]*Window.size[1])*(0.5-self.squareMargin/2) # half of the side length of the line-up square
		midX, midY = Window.size[0]/2, Window.size[1]/2
		# origin is bottom left
		topLeft = np.array(self.movingLayout.to_parent(midX-size, midY+size)) / np.array([Window.size[0], Window.size[1]])
		topRight = np.array(self.movingLayout.to_parent(midX+size, midY+size)) / np.array([Window.size[0], Window.size[1]])
		bottomLeft = np.array(self.movingLayout.to_parent(midX-size, midY-size)) / np.array([Window.size[0], Window.size[1]])
		bottomRight = np.array(self.movingLayout.to_parent(midX+size, midY-size)) / np.array([Window.size[0], Window.size[1]])
		topLeft[1] =1-topLeft[1]
		topRight[1] =1-topRight[1]
		bottomLeft[1] =1-bottomLeft[1]
		bottomRight[1] =1-bottomRight[1]
		return topLeft, topRight, bottomRight, bottomLeft

	def getPosCv(self):
		size = min(self.imgSize[0]*Window.size[0], self.imgSize[1]*Window.size[1])*(0.5-self.squareMargin/2)
		midX, midY = Window.size[0]/2, Window.size[1]/2
		topLeft = self.squareToImg(midX-size, midY+size)
		topRight = self.squareToImg(midX+size, midY+size)
		bottomLeft = self.squareToImg(midX-size, midY-size)
		bottomRight = self.squareToImg(midX+size, midY-size)
		return topLeft, topRight, bottomRight, bottomLeft

	def squareToImg(self, x, y):
		# converts a pos from the movable square to image based position

		kivyPosPx = self.movingLayout.to_parent(x, y) # the position in the window in pixels
		kivyPosPe = np.array(kivyPosPx) / np.array(Window.size) # the position in the window as a percent
		imPos = [ (kivyPosPe[0]-self.imgPos[0])/self.imgSize[0],
		1 - (kivyPosPe[1]-self.imgPos[1])/self.imgSize[1] ] #scaleNumber(kivyPosPe[1], 0, 1, self.imgPos[1], self.imgPos[1]+self.imgSize[1])
		return imPos

	def createImgTexture(self, source = False):
		if source != False:
			self.imgFilename = source
		if self.checkSet:
			# loads image into numpy array
			self.imgNp = image_to_numpy.load_image_file(self.imgFilename).astype(np.uint8) # ImageProcessing.loadImg(self.imgFilename)
			self.imgNp = np.flip(self.imgNp, 0)
			# turn numpy array into buffer
			self.imgBuf = self.imgNp.tostring()
			# then into kivy texture
			self.imgTex = Texture.create(size=(self.imgNp.shape[1], self.imgNp.shape[0]), colorfmt="rgb")
			self.imgTex.blit_buffer(self.imgBuf, bufferfmt="ubyte", colorfmt="rgb") # default colorfmt and bufferfmt
			return True
		else:
			return False

	def makeSquare(self, margin):
		# print("square resizers callback called")
		# print("window size: ", Window.size)

		self.canvas.clear()
		with self.canvas:
			# find if the width/height ratio of the image is more or less than the window
			imgRatio = self.imgNp.shape[1]/self.imgNp.shape[0]
			if (Window.size[0]/Window.size[1]) > (self.imgNp.shape[1]/self.imgNp.shape[0]):
				height = Window.size[1]
				width = height * imgRatio # set the height to fill the window and width to scale according to the image ratio
				self.imgSize = [width/Window.size[0], height/Window.size[1]]
			else:
				width = Window.size[0]
				height = width * (self.imgNp.shape[0]/self.imgNp.shape[1]) # set the width to fill the window and height to scale according to the ratio (backwards)
				self.imgSize = [width/Window.size[0], height/Window.size[1]]
			x = (Window.size[0]/2)-(self.imgSize[0]*Window.size[0])/2 # sets position so that the image is centerd as:
			y = (Window.size[1]/2)-(self.imgSize[1]*Window.size[1])/2 # the middle of the window minus half the image size
			self.imgPos = [x/Window.size[0], y/Window.size[1]]
			Rectangle(pos=(x, y), size=(self.imgSize[0]*Window.size[0], self.imgSize[1]*Window.size[1]), texture = self.imgTex)
			Color(1.0, 0, 0)
			pos = self.getPosKivy()
			for p in pos:
				Line(circle=(p[0], p[1], 25))

			Color(1.0, 1.0, 1.0)

		self.remove_widget(self.movingLayout)
		self.movingLayout.canvas.clear()
		with self.movingLayout.canvas:
			Color(0, 1.0, 0)
			size = min(self.imgSize[0]*Window.size[0], self.imgSize[1]*Window.size[1])*(0.5-margin/2) # half of the side length of the line-up square
			midX, midY = Window.size[0]/2, Window.size[1]/2
			Line(rectangle=(midX-size, midY-size, size*2, size*2), width=3)
		self.add_widget(self.movingLayout)

		self.remove_widget(self.continueButton)
		self.add_widget(self.continueButton)

### 4 ###
class WordsPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.textinput = TextInput(hint_text='Enter words', multiline=False, size_hint = (0.8, 0.1), pos_hint={"x":0.05, "y":0.85} , text_validate_unfocus = False)
		self.textinput.bind(on_text_validate=self.addWord)
		self.add_widget(self.textinput)

		self.addButton = Button(text = "Add Words", size_hint = (0.1, 0.1), pos_hint = {"x":0.85, "y":0.85})
		self.addButton.bind(on_press = self.addWord)
		self.add_widget(self.addButton)

		self.words = []
		self.wordsWidgets = []
		self.wordsLayout = GridLayout(size_hint = (0.9, 0.7), pos_hint = {"x":0.05, "y":0.1})
		self.wordsLayout.cols = 2

		self.add_widget(self.wordsLayout)

		self.continueButton = Button(text="Continue", pos_hint = {"x":0.89, "y":0.01}, size_hint = (0.1, 0.1))
		self.add_widget(self.continueButton)
		self.continueButton.bind(on_press = self.continueToSolve)

		self.img = "not set yet"
		self.cropPos = "not set yet"

	def addWord(self, *args):
		if self.testWord(self.textinput.text):
			self.words.append(self.textinput.text)
			self.textinput.text = ""

			self.wordsWidgets.append(WordWidget(text = self.words[-1], caller = self))
			self.wordsLayout.add_widget(self.wordsWidgets[-1])

	def removeWord(self, wordName):
		wordIndex = self.words.index(wordName)
		self.wordsLayout.remove_widget(self.wordsWidgets[wordIndex])
		del self.words[wordIndex]
		del self.wordsWidgets[wordIndex]

	def continueToSolve(self, *args):
		self.caller.solve_screen.solve(self.img, self.words, self.cropPos)
		self.caller.screen_manager.current = "Solve"

	def getSet(self):
		return self.img == "not set yet" and self.cropPos == "not set yet"

	def setStuff(self, img, pos): # this is the path not the actual image
		self.img = img
		self.cropPos = pos

	def testWord(self, word):
		# tests if you can add a word
		if word == "" or word in self.words:
			return False
		else:
			return True


### 5 ###
class SolvePage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)

		self.againButton = Button(text="Again", pos=(Window.size[0]*0.01, Window.size[1]*0.01), size_hint=(0.1, 0.1))

		with self.canvas:
			self.rect = Rectangle(pos = (0, 0), size=(Window.size[0], Window.size[1]), source = "temp_img.png")
		self.add_widget(self.againButton)
		self.againButton.bind(on_press=self.goAgain)
		# self.imgWidget = Image(source="temp_img.png")
		# self.add_widget(self.rect)
		self.imgPath = "not set"

	def goAgain(self, *args):
		self.caller.screen_manager.current = "Start"

	def setImageBuf(self, img):
		self.imgNp = np.flip(self.img, 0)
		# takes a numpy image
		# turn numpy array into buffer
		self.imgBuf = self.imgNp.tostring()
		# then into kivy texture
		self.imgTex = Texture.create(size=(self.imgNp.shape[1], self.imgNp.shape[0]), colorfmt="rgb")
		self.imgTex.blit_buffer(self.imgBuf, bufferfmt="ubyte", colorfmt="rgb") # default colorfmt and bufferfmt
		self.rect.texture = self.imgTex
		# self.rect.reload()

	def solve(self, imgPath, lookWords, pos):
		self.imgPath = imgPath
		self.img = ImageProcessing.loadImg(self.imgPath)
		grid, gridPlus = ImageProcessing.processImage(self.img, pos, False)
		foundWords = Solvers.wordSearch(grid, lookWords)
		outImg = ImageProcessing.annotate(self.img, gridPlus, pos, foundWords)
		# cv2.imwrite("./result.png", outImg)
		self.setImageBuf(outImg)
		
Builder.load_string('''
<RotatedImage>:
    canvas.before:
        Rotate:
            angle: root.angle
            axis: 0, 0, 1
            origin: root.center
    canvas.after:
        PopMatrix
''')

class RotatedImage(Image):
    angle = NumericProperty()

class SolverApp(App):
	def build(self):
		self.screen_manager = ScreenManager()

		self.start_page = StartPage(self)
		screen = Screen(name = "Start")
		screen.add_widget(self.start_page)
		self.screen_manager.add_widget(screen)

		self.solve_screen = SolvePage(self)
		screen = Screen(name="Solve")
		screen.add_widget(self.solve_screen)
		self.screen_manager.add_widget(screen)

		self.load_screen = LoadPage(self)
		screen = Screen(name="Load")
		screen.add_widget(self.load_screen)
		self.screen_manager.add_widget(screen)

		self.camera_screen = CameraPage(self)
		screen = Screen(name="Camera")
		self.camera_screen.bind(on_pre_enter=self.camera_screen.cameraOn)
		self.camera_screen.bind(on_pre_leave=self.camera_screen.cameraOff)
		screen.add_widget(self.camera_screen)
		self.screen_manager.add_widget(screen)

		self.words_screen = WordsPage(self)
		screen = Screen(name="Words")
		screen.add_widget(self.words_screen)
		self.screen_manager.add_widget(screen)

		self.line_up_screen = LineUpPage(self)
		screen = Screen(name="LineUp")
		screen.add_widget(self.line_up_screen)
		self.screen_manager.add_widget(screen)


		return self.screen_manager

class TestApp(App):
	def build(self):
		self.layout = BoxLayout()
		self.layout.cols = 2
		self.thing = Label(text="test label")
		self.testButton = Button(text="test button")
		self.layout.add_widget(self.thing)
		self.layout.add_widget(self.testButton)
		return self.layout

__version__ = "0.0.1"
if __name__ == "__main__":
	solver_app = SolverApp()
	SolverApp().run()
	# test_app = TestApp()
	# test_app.run()