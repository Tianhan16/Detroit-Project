#!/bin/sh

# Ensure SlidingBrickPuzzle.py exists
if [ ! -f SlidingBrickPuzzle.py ]; then
    echo "Error: SlidingBrickPuzzle.py not found!"
    exit 1
fi

# Check if at least one argument is provided
if [ $# -lt 2 ]; then
    echo "Usage:"
    echo "  ./run.sh print <filename>"
    echo "  ./run.sh done <filename>"
    echo "  ./run.sh availableMoves <filename>"
    echo "  ./run.sh applyMove <filename> \"(piece, 'direction')\""
    echo "  ./run.sh compare <filename1> <filename2>"
    echo "  ./run.sh norm <filename>"
    echo "  ./run.sh random <filename> <steps>"
    echo "  ./run.sh bfs <filename>"
    echo "  ./run.sh dfs <filename>"
    echo "  ./run.sh ids <filename>"
    echo "  ./run.sh astar <filename>"
    echo "  ./run.sh competition <filename>"
    exit 1
fi

# Capture command-line arguments
COMMAND=$1
FILENAME=$2
EXTRA_ARG=$3  # Some commands require extra arguments

# Run the corresponding Python command
case "$COMMAND" in
    print)
        python3 SlidingBrickPuzzle.py print "$FILENAME"
        ;;
    done)
        python3 SlidingBrickPuzzle.py done "$FILENAME"
        ;;
    availableMoves)
        python3 SlidingBrickPuzzle.py availableMoves "$FILENAME"
        ;;
    applyMove)
        if [ -z "$EXTRA_ARG" ]; then
            echo "Error: Missing move argument. Expected \"(piece, 'direction')\""
            exit 1
        fi
        python3 SlidingBrickPuzzle.py applyMove "$FILENAME" "$EXTRA_ARG"
        ;;
    compare)
        if [ -z "$3" ]; then
            echo "Error: Missing second filename for comparison."
            exit 1
        fi
        python3 SlidingBrickPuzzle.py compare "$FILENAME" "$3"
        ;;
    norm)
        python3 SlidingBrickPuzzle.py norm "$FILENAME"
        ;;
    random)
        if [ -z "$EXTRA_ARG" ]; then
            echo "Error: Missing step count for random walk."
            exit 1
        fi
        python3 SlidingBrickPuzzle.py random "$FILENAME" "$EXTRA_ARG"
        ;;
    bfs)
        python3 SlidingBrickPuzzle.py bfs "$FILENAME"
        ;;
    dfs)
        python3 SlidingBrickPuzzle.py dfs "$FILENAME"
        ;;
    ids)
        python3 SlidingBrickPuzzle.py ids "$FILENAME"
        ;;
    astar)
        python3 SlidingBrickPuzzle.py astar "$FILENAME"
        ;;
    competition)
        python3 SlidingBrickPuzzle.py astar "$FILENAME"
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'."
        echo "Usage:"
        echo "  ./run.sh print <filename>"
        echo "  ./run.sh done <filename>"
        echo "  ./run.sh availableMoves <filename>"
        echo "  ./run.sh applyMove <filename> \"(piece, 'direction')\""
        echo "  ./run.sh compare <filename1> <filename2>"
        echo "  ./run.sh norm <filename>"
        echo "  ./run.sh random <filename> <steps>"
        echo "  ./run.sh bfs <filename>"
        echo "  ./run.sh dfs <filename>"
        echo "  ./run.sh ids <filename>"
        echo "  ./run.sh astar <filename>"
        echo "  ./run.sh competition <filename>"
        exit 1
        ;;
esac
