import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.camera import Camera
from kivy.uix.image import Image
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

		self.cameraButton = Button(text = "Camera")
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


		self.imgWidget = Image("temp_img.png")
		self.add_widget(self.imgWidget)
		self.img = "not set"

	def setImage(self, img):
		self.imgWidget = Image("temp_img.png")
		self.add_widget(self.imgWidget)

	def solve(self, img):
		self.img = img
		grid, fourPoint = ImageProcessing.processImage(self.img, False)
		words = Solvers.
		self.setImage(ImageProcessing.annotate(self.img, words, fourPoint))


class LoadPage(GridLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 1
		self.file_thing = FileChooserIconView()
		self.file_thing.bind(on_submit=self.choseFile)
		self.add_widget(self.file_thing)

	def choseFile(self, name, *args):
		print(name)

class CameraPage(FloatLayout):
	def __init__(self, caller, **kwargs):
		self.caller = caller
		super().__init__(**kwargs)
		self.cols = 1

		# self.title = Label(text="camera")
		# self.title.size_hint = (1, 0.1)
		# self.add_widget(self.title)

		self.camera = Camera(play = False)
		self.add_widget(self.camera)
		self.camera.play = True
		self.camera.bind(on_texture=lambda x:print("new frame"))
		self.camera.bind(on_load=lambda x:print("camera started"))

		self.picButton = Button(text = "Take Picture", pos_hint = {"center_x":0.5, "y":0.125}, size_hint = (0.1, 0.1))
		self.picButton.bind(on_press = self.takePictue)
		self.add_widget(self.picButton)

	def takePictue(self, name = "date"):
		img_name = time.strftime("%Y%m%d_%H%M%S")
		self.camera.export_to_png(f"./IMG_{img_name}.png")
		print("Captured "+f"./IMG_{img_name}.png")



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
		screen = Screen(name="Camera (WIP)")
		screen.add_widget(self.camera_screen)
		self.screen_manager.add_widget(screen)


		return self.screen_manager

if __name__ == "__main__":
	solver_app = SolverApp()
	SolverApp().run()