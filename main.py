import kivy
from kivymd.app import MDApp as App
from kivymd.uix.label import MDLabel as Label
from kivymd.uix.gridlayout import MDGridLayout as GridLayout
from kivymd.uix.floatlayout import MDFloatLayout as FloatLayout
from kivymd.uix.button import MDRectangleFlatButton, MDRaisedButton
from kivymd.uix.textfield import MDTextField as TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scatter import Scatter
from kivymd.uix.boxlayout import MDBoxLayout as BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.graphics import *
from kivy.core.window import Window
from kivy.properties import *
from kivy.graphics.texture import Texture
from kivy.utils import platform
from kivy_garden.xcamera import XCamera
from kivymd.uix.progressbar import MDProgressBar
from kivy.clock import Clock
from kivymd.uix.button import MDIconButton
from kivymd.uix.filemanager import MDFileManager
from kivy.clock import mainthread
import time
import numpy as np
import image_to_numpy
from os.path import isfile

from solvers import Solvers
from imageReader import ImageProcessing
from annotator import Annotator

import threading

# kivy.require("2.0")
if platform == "android":
	print("requested permissions")
	from android.permissions import request_permissions, request_permission, Permission
	from android.storage import primary_external_storage_path
	request_permissions([Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE, Permission.CAMERA])

### step 1 ###
class StartPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)

		self.title = Label(text = "Find-a-word Solver")
		self.title.size_hint = (1, 0.1)
		self.title.pos_hint = {"center_x":0.5, "y":0.9}
		self.title.halign = "center"
		self.add_widget(self.title)

		self.buttons = FloatLayout(adaptive_size=True)
		self.buttons.cols = 2

		self.loadButton = MDRaisedButton(text="Load File")
		self.loadButton.size_hint = (0.9, 0.4)
		self.loadButton.pos_hint = {"x":0.05, "y":0.05}
		self.loadButton.bind(on_release = self.loadImage)
		self.add_widget(self.loadButton)

		self.cameraButton = MDRaisedButton(text = "Camera (WIP)")
		self.cameraButton.size_hint = (0.9, 0.4)
		self.cameraButton.pos_hint = {"x":0.05, "y":0.5}
		self.cameraButton.bind(on_release = self.launchCamera)
		self.add_widget(self.cameraButton)

		# self.add_widget(self.buttons)

	def loadImage(self, instance):
		self.title.text = "test"
		self.caller.goToPage("Load")

	def launchCamera(self, instance):
		self.caller.goToPage("Camera")

### step 2, load###
class LoadPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		print("caller goToPage", type(self.caller.goToPage))
		super().__init__()
		self.file_thing = MDFileManager()
		self.file_thing.preview = True
		self.file_thing.select_path = self.select_path
		if platform == "android":
			from os.path import join
			print("primary_external_storage_path : ", join(primary_external_storage_path(), "DCIM", "Camera"))
			self.path = join(primary_external_storage_path(), "DCIM", "Camera")
		else:
			self.path =  "/home/olikat/word-search-solver/test_images/fulls"
		# self.add_widget(self.file_thing)

	def select_path(self, path, *args):
		if isfile(path):
			print("thing selected", path)
			self.file_thing.close()

			self.caller.pages["LineUp"].setImage(path, camera = False)
			self.caller.goToPage("LineUp")
		else:
			print("non file selected", path)

	def stopped(self):
		self.file_thing.close()

	def started(self):
		self.file_thing.show(self.path)

### step 2, camera ###
class CameraPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)

		self.camera = XCamera(play = True)
		self.camera.on_picture_taken = self.picture_taken
		self.add_widget(self.camera)
		if platform == "android":
			from os.path import join
			self.camera.directory = join(primary_external_storage_path(), "DCIM", "Camera")

		self.title = Label(text="Try to get the grid flat and square-on")
		self.title.size_hint = (1, 0.1)
		self.add_widget(self.title)

	def picture_taken(self, filename):
		while not isfile(filename):
			print("waiting for file to save")
			time.sleep(0.1)
		self.leave(filename)

	def leave(self, filename):
		self.caller.pages["LineUp"].setImage(filename, camera = True)
		print('Picture taken and saved to {}'.format(filename))
		self.caller.goToPage("LineUp")

	def started(self):
		self.camera.force_landscape()

	def stopped(self):
		self.camera.restore_orientation()

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

		self.continueButton = MDRaisedButton(text="Continue", size_hint = (0.15, 0.1), pos_hint = {"x":0.85, "y":0.85})
		self.add_widget(self.continueButton)
		self.continueButton.bind(on_release = self.continued)
		self.on_touch_up = self.buttonTouchCheck

		self.squareMargin = 0.1
		self.createImgTexture()
		Window.bind(on_resize=lambda *args:self.makeSquare(self.squareMargin))
		# self.movingLayout.bind(on_transform_with_touch=lambda *args:print(self.getPosCv()))
		self.makeSquare(self.squareMargin)

	def buttonTouchCheck(self, touch, *args):
		# print("Widget Pos", self.continueButton.pos, self.continueButton.size)
		# print("Touch pos ", self.to_local(touch.pos[0], touch.pos[1]))
		# print("Touch Check ", self.continueButton.collide_point(touch.pos[0], touch.pos[1]))
		if self.continueButton.collide_point(*touch.pos):
			self.continued()
		else:	
			return self.movingLayout.on_touch_up(touch)

	def setImage(self, img, camera = False):
		print("path given: ",img)
		self.imgFilename = img
		self.createImgTexture(img)
		self.makeSquare(self.squareMargin)

	def checkSet(self):
		return self.imgFilename != "temp_img.png"

	def continued(self, *args):
		print(self.getPosCv())
		if self.checkSet:
			self.caller.pages["Words"].setStuff(self.imgFilename, self.getPosCv())
			self.caller.goToPage("Words")

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

	def createImgTexture(self, source = "temp_img.png"):
		if source != "temp_img.png":
			print("source for createImgTexture was", source)
			self.imgFilename = source
		else:
			print("didnt get source for createImgTexture")
		# loads image into numpy array
		self.imgNp = image_to_numpy.load_image_file(source).astype(np.uint8)
		self.imgNp = np.flip(self.imgNp, 0)
		# turn numpy array into buffer
		self.imgBuf = self.imgNp.tobytes()
		# then into kivy texture
		self.imgTex = Texture.create(size=(self.imgNp.shape[1], self.imgNp.shape[0]), colorfmt="rgb")
		self.imgTex.blit_buffer(self.imgBuf, bufferfmt="ubyte", colorfmt="rgb") # default colorfmt and bufferfmt

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

		self.addButton = MDRaisedButton(text = "Add Words", size_hint = (0.1, 0.1), pos_hint = {"x":0.85, "y":0.85})
		self.addButton.bind(on_release = self.addWord)
		self.add_widget(self.addButton)

		self.words = []
		self.wordsWidgets = []
		self.wordsLayout = GridLayout(size_hint = (0.9, 0.7), pos_hint = {"x":0.05, "y":0.1})
		self.wordsLayout.cols = 2

		self.add_widget(self.wordsLayout)

		self.continueButton = MDRaisedButton(text="Continue", pos_hint = {"x":0.89, "y":0.01}, size_hint = (0.1, 0.1))
		self.add_widget(self.continueButton)
		self.continueButton.bind(on_release = self.continueToSolve)

		self.img = "not set yet"
		self.cropPos = "not set yet"

	def addWord(self, *args):
		if self.testWord(self.textinput.text):
			self.words.append(self.textinput.text)
			self.textinput.text = ""

			self.wordsWidgets.append(WordWidget(text = self.words[-1], caller = self, appCaller = self.caller))
			self.wordsLayout.add_widget(self.wordsWidgets[-1])

	def removeWord(self, wordName):
		wordIndex = self.words.index(wordName)
		self.wordsLayout.remove_widget(self.wordsWidgets[wordIndex])
		del self.words[wordIndex]
		del self.wordsWidgets[wordIndex]

	def continueToSolve(self, *args):
		Clock.schedule_once(lambda *_:self.caller.pages["Solver"].solve(self.img, self.words, self.cropPos))
		self.caller.goToPage("Solver")

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

class WordWidget(RelativeLayout):
	def __init__(self, text, caller, appCaller, **kwargs):
		# self.radius = [25, ]
		# self.md_bg_color = appCaller.theme_cls.primary_color
		# icons: "delete" "delete-circle" "delete-forever" close" "close-circle" "close-cricle-outline"
		self.caller = caller
		super().__init__(**kwargs)
		self.textLabel = Label(text=text)
		self.add_widget(self.textLabel)

		self.removeButton = MDIconButton(icon="delete")
		self.removeButton.user_font_size = "32sp"
		self.removeButton.theme_text_color = "Custom"
		self.removeButton.text_color = appCaller.theme_cls.primary_color
		self.removeButton.pos_hint = {"center_x":0.4, "center_y":0.5}
		self.removeButton.bind(on_release=self.remove)
		self.add_widget(self.removeButton)

	def remove(self, *args):
		self.caller.removeWord(self.textLabel.text)

### 5 ###
class SolvePage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)

		self.loader = MDProgressBar()
		self.loader.pos_hint = {"center_x":0.5, "center_y":0.55}
		self.add_widget(self.loader)

		self.message = Label(text = "Loading")
		self.message.pos_hint = {"center_x":0.5, "center_y":0.4}
		self.add_widget(self.message)

	@mainthread
	def setLoadInfo(self, value, message):
		print("Old", self.loader.value, "	New", (self.loader.value+value))
		self.loader.value += value
		self.message.text = message

	def solve(self, imgPath, lookWords, pos):
		# call _solve in another thread
		self.outImg = "not set"
		self.thread = threading.Thread(target=SolvePage._solve, args=(imgPath, lookWords, pos, self))
		print("solve looking for", lookWords, "in", imgPath, "at", pos)
		self.checkDoneEvent = Clock.schedule_interval(self.checkDone, 1) # checks if _solve is done and goes to next page
		self.thread.start()

	def _solve(imgPath, words, pos, pageInst):
		pageInst.setLoadInfo(0, "Loading Image")
		pageInst.imgPath = imgPath
		pageInst.img = ImageProcessing.loadImg(pageInst.imgPath)
		grid, gridPlus, allGrids = ImageProcessing.processImage(pageInst.img, pos, debug = False, progressCallback = pageInst.setLoadInfo)
		pageInst.setLoadInfo(10, "Finding Words")
		foundWords = Solvers.wordSearch(allGrids, words, True)
		pageInst.setLoadInfo(10, "Annotating Image")
		outImg = Annotator.annotate(pageInst.img, gridPlus, pos, foundWords)
		pageInst.setLoadInfo(10, "Done")
		pageInst.outImg = outImg

	@mainthread
	def done(self, outImg):
		self.checkDoneEvent.cancel()
		print("called done")
		self.thread.join()
		self.caller.pages["Final"].setImageBuf(self.outImg)
		self.caller.goToPage("Final")

	@mainthread
	def checkDone(self, *args):
		if self.outImg == "not set":
			return False
		else:
			print("outImg has been set")
			self.done(self.outImg)

	def stopped(self):
		pass
		# print("got to stopped")
		# self.thread.join()
		# self.done(self.outImg)
		# print("got past .join")

### 6 ###
class FinalPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)

		self.againButton = MDRaisedButton(text="Again", pos=(Window.size[0]*0.01, Window.size[1]*0.01), size_hint=(0.1, 0.1))

		with self.canvas:
			self.rect = Rectangle(pos = (0, 0), size=(Window.size[0], Window.size[1]))
		self.add_widget(self.againButton)
		self.againButton.bind(on_release=self.goAgain)
		self.imgPath = "not set"

	def goAgain(self, *args):
		self.caller.goToPage("Start")

	def setImageBuf(self, img):
		print("final page image set ", type(img), img.shape)
		self.imgNp = np.flip(img, 0)
		# takes a numpy image
		# turn numpy array into buffer
		self.imgBuf = self.imgNp.tobytes()
		# then into kivy texture
		self.imgTex = Texture.create(size=(self.imgNp.shape[1], self.imgNp.shape[0]), colorfmt="rgb")
		self.imgTex.blit_buffer(self.imgBuf, bufferfmt="ubyte", colorfmt="rgb") # default colorfmt and bufferfmt

		self.canvas.clear()
		self.remove_widget(self.againButton)
		with self.canvas:
			self.rect = Rectangle(pos = (0, 0), size = (Window.size[0], Window.size[1]), colorfmt="rgb", texture = self.imgTex)
		self.add_widget(self.againButton)
		
class SolverApp(App):
	def addPage(self, name, pageClass):
		self.pages[name] = pageClass(self)
		screen = Screen(name=name)
		screen.add_widget(self.pages[name])
		self.screen_manager.add_widget(screen)

	def build(self):
		self.screen_manager = ScreenManager()
		self.lastPages = []
		self.pages = {}

		self.addPage("Start", StartPage)

		self.addPage("Load", LoadPage)

		self.addPage("Camera", CameraPage)

		self.addPage("LineUp", LineUpPage)

		self.addPage("Words", WordsPage)

		self.addPage("Solver", SolvePage)

		self.addPage("Final", FinalPage)

		return self.screen_manager

	def goToPage(self, name):
		print(f"going to {name} page")
		self.lastPages.append(self.screen_manager.current)
		self.screen_manager.current = name

		try:
			self.pages[self.lastPages[-1]].stopped()
		except AttributeError:
			print("no start function")

		try:
			self.pages[name].started()
		except AttributeError:
			print("no end function")

	def backPage(self):
		if len(self.lastPages) >= 1:
			self.screen_manager.current = self.lastPages[-1]
			del self.lastPages[-1]


class TestApp(App):
	def build(self):
		import time
		time.sleep(5)
		self.layout = FloatLayout()
		try:
			self.camera = XCamera(play = True)
			# self.camera.play = False
			self.layout.add_widget(self.camera)
		except AttributeError:
			print("XCamera failed to start")
		
		self.testButton = MDRaisedButton(text="test button")
		self.layout.add_widget(self.testButton)
		return self.layout

__version__ = "0.1"
if __name__ == "__main__":
	solver_app = SolverApp()
	SolverApp().run()
	# test_app = TestApp()
	# test_app.run()