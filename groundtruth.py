import numpy as np

class GroundTruth():
	def __init__(self, filepath):
		self.filepath = filepath
		self.data = {}
		self.labels = []

		self.type_of_activities = ["normal", "physical"]
		self.motion_extraction_position = [2, 12]
		self.threshold_levels = [0.06, 0.08, 0.07, 0.10]
		self.sampling_rate = 250
		self.recorded_time_duration = 20

		self.process_labels()
		self.process_data()

	def process_labels(self):
		for activity_type in self.type_of_activities:
			for x in range(1,16):
				self.labels.append("p" + str(x) + "_" + activity_type)
		
	def process_data(self):
		ground_truth_data = list(np.loadtxt(self.filepath, str, delimiter='\n'))
		for detail in ground_truth_data:
			detail = detail.split(",")
			try:
				value = float(detail[1])
				self.data[detail[0]] = value
			except (ValueError, TypeError):
				pass