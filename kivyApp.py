# import logging
# logging.getLogger("kivy").setLevel(logging.ERROR)
# logging.getLogger("keras").setLevel(logging.ERROR)
import os
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
import time
import numpy as np
import image_to_numpy

from solvers import Solvers
from imageReader import ImageProcessing
kivy.require("1.11.1")

### 1 ###
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

### 5 ###
class SolvePage(GridLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 1
		self.add_widget(Label(text="LETS GOOOO!!!!"))


		self.imgWidget = Image(source="temp_img.png")
		self.add_widget(self.imgWidget)
		self.imgPath = "not set"

	def setImage(self, img):
		self.imgWidget.source = "temp_img.png"
		self.imgWidget.reload()

	def solve(self, imgPath, words, pos):
		self.imgPath = imgPath
		self.img = ImageProcessing.loadImg(self.imgPath)
		grid, letters, _ = ImageProcessing.processImage(self.img, pos, False)
		print(grid)
		words = Solvers.wordSearch(grid, words)
		self.setImage(ImageProcessing.annotate(self.img, words, pos))

### 2.1 ###
class LoadPage(GridLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 1
		self.file_thing = FileChooserIconView()
		self.file_thing.bind(on_submit=self.choseFile)
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

### 2.2 ###
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

Builder.load_string('''
<RotatedImage>:
    canvas.before:
        PushMatrix
        Rotate:
            angle: root.angle
            axis: 0, 0, 1
            origin: root.center
    canvas.after:
        PopMatrix
''')

class RotatedImage(Image):
    angle = NumericProperty()

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

		self.squareMargin = 0.1
		# self.bind(on_size=lambda _:self.makeSquare(), on_pos=lambda _:self.makeSquare(self.squareMargin))
		Window.bind(on_resize=lambda *args:self.makeSquare(self.squareMargin))
		self.createImgTexture()
		self.makeSquare(self.squareMargin)

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
		size = min(Window.size[0], Window.size[1])*(0.5-self.squareMargin/2) # half of the side length of the line-up square
		midX, midY = Window.size[0]/2, Window.size[1]/2
		topLeft = self.movingLayout.to_parent(midX-size, midY+size)
		topRight = self.movingLayout.to_parent(midX+size, midY+size)
		bottomLeft = self.movingLayout.to_parent(midX-size, midY-size)
		bottomRight = self.movingLayout.to_parent(midX+size, midY-size)
		return topLeft, topRight, bottomRight, bottomLeft

	def getPosCv(self):
		# gets the coordinates of each corner of the line-up square with the origin being top left as a percentage of the window size
		size = min(Window.size[0], Window.size[1])*(0.5-self.squareMargin/2) # half of the side length of the line-up square
		midX, midY = Window.size[0]/2, Window.size[1]/2
		# origin is bottom left
		topLeft = np.array(self.movingLayout.to_parent(midX-size, midY-size)) / np.array([Window.size[0], Window.size[1]])
		topRight = np.array(self.movingLayout.to_parent(midX+size, midY-size)) / np.array([Window.size[0], Window.size[1]])
		bottomLeft = np.array(self.movingLayout.to_parent(midX-size, midY+size)) / np.array([Window.size[0], Window.size[1]])
		bottomRight = np.array(self.movingLayout.to_parent(midX+size, midY+size)) / np.array([Window.size[0], Window.size[1]])
		return topLeft, topRight, bottomRight, bottomLeft

	def createImgTexture(self, source = False):
		if source != False:
			self.imgFilename = source
		if self.checkSet:
			# loads image into numpy array
			self.imgNp = ImageProcessing.loadImg(self.imgFilename).astype(np.uint8)
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
		print("square resizers callback called")
		print("window size: ", Window.size)

		self.canvas.clear()
		with self.canvas:
			Rectangle(pos=(0, 0), size=(Window.size[0], Window.size[1]), texture = self.imgTex)
			Color(1.0, 0, 0)
			pos = self.getPosKivy()
			for p in pos:
				Line(circle=(p[0], p[1], 25))

			Color(1.0, 1.0, 1.0)

		self.remove_widget(self.movingLayout)
		self.movingLayout.canvas.clear()
		with self.movingLayout.canvas:
			print(Window.size[0], Window.size[1])
			Color(0, 1.0, 0)
			size = min(Window.size[0], Window.size[1])*(0.5-margin/2) # half of the side length of the line-up square
			midX, midY = Window.size[0]/2, Window.size[1]/2
			Line(rectangle=(midX-size, midY-size, size*2, size*2))
		self.add_widget(self.movingLayout)

		self.remove_widget(self.continueButton)
		self.add_widget(self.continueButton)



class SolverApp(App):
	def build(self):
		print(self)
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

if __name__ == "__main__":
	solver_app = SolverApp()
	SolverApp().run()