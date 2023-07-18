from wiki_trace import WikiTrace

file_dir = './dataset/'
filename = 'wiki2018_dev.tr'

with WikiTrace(f"{file_dir}{filename}", max_look_ahead=100) as trace:
    while not trace.done():
        time, obj_id, obj_size, obj_type = trace.next()
        print(f"object {obj_id} is requested at {time}.")
