import os

class ExperimentLogger:
    def __init__(self, log_file):
        # Ensure the directory exists
        directory = os.path.dirname(log_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        self.log_file = log_file
        self.data = []

    def log(self, layout, pacman, numGames, scores, times, wins, SCARED_TIME, numSimulations):
        average_score = sum(scores) / float(len(scores))
        average_time = sum(times) / float(len(times))
        winRate = (wins.count(True) / len(wins)) * 100

        record = (
            f"Maze Name:          {layout}\n"
            f"Pacman Agent Type:  {pacman}\n"
            f"Number of Games:    {numGames}\n"
            f"Average Score:      {average_score:.2f}\n"
            f"Scores:             {', '.join([str(score) for score in scores])}\n"
            f"Times:              {', '.join([str(round(time, 2)) for time in times])}\n"
            f"Average Time:       {round(average_time, 2)}\n"
            f"Win Rate:           {wins.count(True)}/{len(wins)} ({winRate:.2f}%)\n"
            f"Record:             {', '.join([['Loss', 'Win'][int(w)] for w in wins])}\n"
            f"Scared Time:        {SCARED_TIME}\n"
            f"Num Simulations:    {numSimulations}\n"
            "----------------------------------------\n"
        )

        self.data.append(record)
        self.save_log(record)

    def save_log(self, record):
        with open(self.log_file, 'a') as file:
            file.write(record)

    def save_summary(self):
        with open(self.log_file, 'a') as file:
            for record in self.data:
                file.write(record)