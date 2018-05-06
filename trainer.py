import numpy as np
import referee_trainer
def main():
    p1 = 0
    p2 = 0
    draws = 0
    player1 = "playerw"
    #player1= "playerLearner"
    player2 = "player"
    for x in range(100):
        if np.random.randint(0,2) == 1:
            winner = referee_trainer.main(player1, player2)
            if (winner == "B"):
                print(player2)
                p2+=1
            elif (winner == "W"):
                print(player1)
                p1+=1
            else:
                print("Draw")
                draws+=1
        else:
            winner = referee_trainer.main(player2, player1)
            if (winner == "W"):
                print(player2)
                p2+=1
            elif (winner == "B"):
                print(player1)
                p1+=1
            else:
                print("Draw")
                draws+=1

    print(player1, ": ", p1)
    print(player2, ": ", p2)
    print("Draws: ", draws)

main()