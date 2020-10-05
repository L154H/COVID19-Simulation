class Output:
    """
    stores all events to analyse or print them later
    """

    def __init__(self):
        self.events = []

    def add_event(self, event):
        self.events.append(event)

    def print_output(self):
        print(self.events)
        print("Number of Events:")
        print(len(self.events))
