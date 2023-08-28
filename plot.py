import matplotlib.pyplot as plt
import pandas as pd
import funcs


# График уменьшения количества "успехов" по мере удаления от Москвы

try:
    success_tst = dict()
    with open('data/plot_1_data.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            my_tuple = line.replace('(', '').replace(')', '').split(', ')
            success_tst[eval(my_tuple[0])] = eval(my_tuple[1])

    success_sorted_list = sorted(success_tst, reverse=True)

    success_sorted = dict()
    for entry in success_sorted_list:
        success_sorted[entry] = success_tst[entry]

except FileNotFoundError:

    df = pd.read_csv('data/ga_hits.csv')
    df2 = pd.read_csv('data/ga_sessions.csv')
    df5 = pd.merge(df, df2, on='session_id')
    df = None
    df2 = None

    df5 = funcs.event_action(df5)
    df5 = funcs.distance_from_moscow(df5)
    df5 = df5[['event_action', 'distance_from_moscow']]

    success = df5[df5['event_action'] == 1].value_counts()
    no_success = df5[df5['event_action'] == 0].value_counts()

    success = dict(success[1])
    no_success = dict(no_success[0])

    for entry in no_success.keys():
        if entry in success.keys():
            continue
        else:
            success[entry] = 0

    success_sorted_list = sorted(success, reverse=True)

    success_sorted = dict()
    for entry in success_sorted_list:
        success_sorted[entry] = success[entry]

    success_list = list(zip(success.keys(), success.values()))

    with open('data/plot_1_data.txt', 'w') as file:
        for entry in success_list:
            line = f"{entry}\n"
            file.write(line)

finally:
    x = success_sorted.keys()
    y = success_sorted.values()

    plt.plot(x, y)

    plt.xlabel('удаленность от Москвы')
    plt.ylabel('Количество "Успехов"')

    plt.title('График 1')

    plt.show()
