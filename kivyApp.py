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
from kivy.clock import Clock
from kivy.graphics import *
import time

from solvers import Solvers
from imageReader import ImageProcessing
kivy.require("1.11.1")

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

class SolvePage(GridLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 1
		self.add_widget(Label(text="LETS GOOOO!!!!"))


		self.imgWidget = Image(source="temp_img.png")
		self.add_widget(self.imgWidget)
		self.img = "not set"

	def setImage(self, img):
		self.imgWidget = Image("temp_img.png")
		self.add_widget(self.imgWidget)

	def solve(self, imgPath, words):
		self.img = img
		grid, fourPoint = ImageProcessing.processImage(self.img, False)
		words = Solvers.wordSearch(grid, words)
		self.setImage(ImageProcessing.annotate(self.img, words, fourPoint))


class LoadPage(GridLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 1
		self.file_thing = FileChooserIconView()
		self.file_thing.bind(on_submit=self.choseFile)
		self.add_widget(self.file_thing)

	def choseFile(self, x, *args):
		self.caller.screen_manager.current = "LineUp"
		self.caller.line_up_screen.setImage(x.selection[0])
		# print("chosen file: ", x.path, x.selection)

class WordsPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.calller = caller
		super().__init__(**kwargs)
		self.cols = 1
		self.textinput = TextInput(hint_text='Enter words', multiline=False, size_hint = (0.8, 0.1), pos_hint={"x":0.05, "y":0.85} , text_validate_unfocus = False)
		self.textinput.bind(on_text_validate=self.addWord)
		self.add_widget(self.textinput)

		self.addButton = Button(text = "Add Words", size_hint = (0.1, 0.1), pos_hint = {"x":0.85, "y":0.85})
		self.addButton.bind(on_press = self.addWord)
		self.add_widget(self.addButton)

		self.words = []
		self.wordsWidgets = []
		self.wordsLayout = GridLayout(size_hint = (0.9, 0.7), pos_hint = {"x":0.05, "y":0.05})
		self.wordsLayout.cols = 2

		self.add_widget(self.wordsLayout)

		self.continueButton = Button(text="Continue", pos_hint = {"x":0.89, "y":0.01}, size_hint = (0.1, 0.1))
		self.add_widget(self.continueButton)

		self.img = "not set yet"

	def addWord(self, *args):
		self.words.append(self.textinput.text)
		self.textinput.text = ""

		self.wordsWidgets.append(Label(text = self.words[-1])) #  pos_hint={"center_x": 0.05, "center_y":0.675-len(self.words)*0.04})
		self.wordsLayout.add_widget(self.wordsWidgets[-1])

	def continueToSolve(self, *args):
		self.caller.solve_screen.solve(self.img, self.words)
		self.caller.screen_manager.current = "Solver"

	def setImage(self, img): # this is the path not the actual image
		self.img = img

class CameraPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)

		self.camera = Camera(play = True)
		self.camera.play = True
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
		Clock
		print("Captured "+f"./IMG_{img_name}.png")

class LineUpPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)

		self.movingLayout = Scatter()
		self.movingLayout.auto_bring_to_front = False

		self.imgWidget = Image(source="temp_img.png")

		self.movingLayout.add_widget(self.imgWidget)

		self.add_widget(self.movingLayout)

		self.bind(on_size=lambda _:self.makeSquare(0.1), on_pos=lambda _:self.makeSquare(0.1))
		self.makeSquare()

	def setImage(self, img):
		print("path given: ",img)
		self.imgWidget.source = img
		self.imgWidget.reload()

	def checkSet(self):
		return self.imgWidget.source == "temp_img.png"
		
	def makeSquare(self, margin=0.1):
		with self.canvas:
			Color(0, 1.0, 0)
			print(self.width, self.height)
			size = min(self.width, self.height)*(0.5-margin/2) # half of the side length of the line-up square
			midX, midY = self.width/2, self.height/2
			print("\n\n\n self", midX, midY, "\n\n\n")
			Line(rectangle=(midX-size, midY-size, size*2, size*2))


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