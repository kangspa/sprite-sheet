def test(**kwargs):
    num, stair = None, None
    for k, v in kwargs.items():
        if k == "num": num = v
        elif k == "stair": stair = v
    print(num,stair)
test(num=7, stair=6)
test(stair=2)

def sest(*args):
    for val in args:
        print(val)
sest(3,4,2)
sest(1,4,'ets',[142,321,14552])