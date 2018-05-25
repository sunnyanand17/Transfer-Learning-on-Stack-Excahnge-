import pandas as pd


untagged = 0
cooking_counts = {}
data = pd.read_csv("cooking.csv")
cooking_tags = set()
for tag in data['tags']:
    question_tags = tag.split(' ')
    for t in question_tags:
        cooking_tags.add(t)
        if t == "untagged":
            untagged += 1
        if t in cooking_counts:
            cooking_counts[t] += 1
        else:
            cooking_counts[t] = 1

print "cooking count: " + str(len(cooking_tags))
cooking_unique_tags = 0
for tag in cooking_counts:
    if cooking_counts[tag] == 1:
        cooking_unique_tags += 1
print "cooking unique tags: " + str(cooking_unique_tags)


biology_counts = {}
data = pd.read_csv("biology.csv")
biology_tags = set()
for tag in data['tags']:
    question_tags = tag.split(' ')
    for t in question_tags:
        
        biology_tags.add(t)
        if t == "untagged":
            untagged += 1
        if t in biology_counts:
            biology_counts[t] += 1
        else:
            biology_counts[t] = 1

print "biology count: " + str(len(biology_tags))
biology_unique_tags = 0
for tag in biology_counts:
    if biology_counts[tag] == 1:
        biology_unique_tags += 1
print "biology unique tags: " + str(biology_unique_tags)


crypto_counts = {}
data = pd.read_csv("crypto.csv")
crypto_tags = set()
for tag in data['tags']:
    question_tags = tag.split(' ')
    for t in question_tags:
        crypto_tags.add(t)
        if t == "untagged":
            untagged += 1
        if t in crypto_counts:
            crypto_counts[t] += 1
        else:
            crypto_counts[t] = 1

print "crypto count: " + str(len(crypto_tags))
crypto_unique_tags = 0
for tag in crypto_counts:
    if crypto_counts[tag] == 1:
        crypto_unique_tags += 1
print "crypto unique tags: " + str(crypto_unique_tags)


diy_counts = {}
data = pd.read_csv("diy.csv")
diy_tags = set()
for tag in data['tags']:
    question_tags = tag.split(' ')
    for t in question_tags:
        diy_tags.add(t)
        if t == "untagged":
            untagged += 1
        if t in diy_counts:
            diy_counts[t] += 1
        else:
            diy_counts[t] = 1

print "diy count: " + str(len(diy_tags))
diy_unique_tags = 0
for tag in diy_counts:
    if diy_counts[tag] == 1:
        diy_unique_tags += 1
print "diy unique tags: " + str(diy_unique_tags)



robotics_counts = {}
data = pd.read_csv("robotics.csv")
robotics_tags = set()
for tag in data['tags']:
    question_tags = tag.split(' ')
    for t in question_tags:
        robotics_tags.add(t)
        if t == "untagged":
            untagged += 1
        if t in robotics_counts:
            robotics_counts[t] += 1
        else:
            robotics_counts[t] = 1

print "robotics count: " + str(len(robotics_tags))
robotics_unique_tags = 0
for tag in robotics_counts:
    if robotics_counts[tag] == 1:
        robotics_unique_tags += 1
print "robotics unique tags: " + str(robotics_unique_tags)


travel_counts = {}
data = pd.read_csv("travel.csv")
travel_tags = set()
for tag in data['tags']:
    question_tags = tag.split(' ')
    for t in question_tags:
        travel_tags.add(t)
        if t == "untagged":
            untagged += 1
        if t in travel_counts:
            travel_counts[t] += 1
        else:
            travel_counts[t] = 1

print "travel count: " + str(len(travel_tags))
travel_unique_tags = 0
for tag in travel_counts:
    if travel_counts[tag] == 1:
        travel_unique_tags += 1
print "travel unique tags: " + str(travel_unique_tags)
all_unique_tags = set.union(cooking_tags, biology_tags, crypto_tags, diy_tags, robotics_tags, travel_tags)
print set.intersection(cooking_tags, biology_tags, crypto_tags, diy_tags, robotics_tags, travel_tags)
print "total unique tags: " + str(len(all_unique_tags))
print "total untagged questions: " + str(untagged)
all_tags = [cooking_tags, biology_tags, crypto_tags, diy_tags, robotics_tags, travel_tags]

for topic_a in all_tags:
    for topic_b in all_tags:
        print len(set.intersection(topic_a, topic_b))


