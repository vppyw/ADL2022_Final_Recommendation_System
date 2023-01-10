import json

courses = json.load(open("../hahow/courses.json", "r"))
feature_map = json.load(open("./course_feature_map.json", "r"))
COURSE_FEATURE_NUM = 251
# NO_GROUP_FEATURE = 252?
PRICE_TAG = 252 # 252 ~ 257 0: free, 1: <= 1000, 2: <= 2000, 3: <= 3000, 4: <= 4000, 5: > 4000,
TEACHER_ID_START = 258
course_data = []

# record the teacher teaching more than one courses
tmp_set, multi_teacher = set(), set()
teacher_feature = {}
for c in courses:
    tid = c["teacher_id"] 
    if tid in tmp_set:
        multi_teacher.add(tid)
    tmp_set.add(tid)

for idx, tid in enumerate(multi_teacher):
    teacher_feature[tid] = idx

for idx, course in enumerate(courses):
    data = {}
    data["course_id"] = course["course_id"]
    data["unique_id"] = idx + 1
    feature = data["feature"] = [0] * (COURSE_FEATURE_NUM)
    # Course Feature
    for group in course["groups"]:
        if group != "":
            feature[ feature_map[group]-1 ] = feature_map[group]
    for sub_group in course["sub_groups"]:
        if sub_group != "":
            feature[ feature_map[sub_group]-1 ] = feature_map[sub_group]
    for topic in course["topics"]:
        if topic != "":
            feature[ feature_map[topic]-1 ] = feature_map[topic]

    # Price
    price = PRICE_TAG + min( int(course["course_price"]) // 1000, 5 )    
    data["feature"].append(price)
    
    # Teacher ID
    data["feature"].append( TEACHER_ID_START + teacher_feature.get( course["teacher_id"], -TEACHER_ID_START ))
    course_data.append(data)

print(TEACHER_ID_START + len(teacher_feature))
json.dump( course_data, open("./course_with_feature_price_teacher.json", "w"), indent = 2 )
