"""
"""
import csv
import json
import math
import multiprocessing.dummy
import os
import random

import numpy as np
import skimage.draw
import skimage.io
import tensorflow as tf

import qdraw.dataset as dataset

num_train_csv_lines = {
    "airplane.csv": 126847,
    "alarm_clock.csv": 104699,
    "ambulance.csv": 125383,
    "angel.csv": 119183,
    "animal_migration.csv": 107590,
    "ant.csv": 106422,
    "anvil.csv": 108163,
    "apple.csv": 130265,
    "arm.csv": 103350,
    "asparagus.csv": 137543,
    "axe.csv": 106213,
    "backpack.csv": 108681,
    "banana.csv": 276723,
    "bandage.csv": 117116,
    "barn.csv": 120594,
    "baseball_bat.csv": 106001,
    "baseball.csv": 117093,
    "basketball.csv": 116947,
    "basket.csv": 99387,
    "bat.csv": 87942,
    "bathtub.csv": 143638,
    "beach.csv": 106281,
    "bear.csv": 118171,
    "beard.csv": 137235,
    "bed.csv": 97619,
    "bee.csv": 101722,
    "belt.csv": 160342,
    "bench.csv": 110860,
    "bicycle.csv": 112845,
    "binoculars.csv": 99993,
    "bird.csv": 103181,
    "birthday_cake.csv": 127802,
    "blackberry.csv": 104885,
    "blueberry.csv": 103468,
    "book.csv": 101839,
    "boomerang.csv": 126042,
    "bottlecap.csv": 122740,
    "bowtie.csv": 114421,
    "bracelet.csv": 99139,
    "brain.csv": 117433,
    "bread.csv": 107244,
    "bridge.csv": 112058,
    "broccoli.csv": 117116,
    "broom.csv": 103518,
    "bucket.csv": 111725,
    "bulldozer.csv": 156669,
    "bus.csv": 135580,
    "bush.csv": 102114,
    "butterfly.csv": 104825,
    "cactus.csv": 115550,
    "cake.csv": 102899,
    "calculator.csv": 111092,
    "calendar.csv": 290761,
    "camel.csv": 105903,
    "camera.csv": 115002,
    "camouflage.csv": 142234,
    "campfire.csv": 115721,
    "candle.csv": 125643,
    "cannon.csv": 110986,
    "canoe.csv": 109473,
    "car.csv": 151981,
    "carrot.csv": 116001,
    "castle.csv": 107153,
    "cat.csv": 94660,
    "ceiling_fan.csv": 98723,
    "cello.csv": 119283,
    "cell_phone.csv": 103775,
    "chair.csv": 194406,
    "chandelier.csv": 136872,
    "church.csv": 133645,
    "circle.csv": 109140,
    "clarinet.csv": 108452,
    "clock.csv": 108799,
    "cloud.csv": 107835,
    "coffee_cup.csv": 152639,
    "compass.csv": 111504,
    "computer.csv": 108019,
    "cookie.csv": 116952,
    "cooler.csv": 240358,
    "couch.csv": 102626,
    "cow.csv": 92850,
    "crab.csv": 98105,
    "crayon.csv": 110634,
    "crocodile.csv": 98119,
    "crown.csv": 120131,
    "cruise_ship.csv": 103050,
    "cup.csv": 111873,
    "diamond.csv": 117242,
    "dishwasher.csv": 139066,
    "diving_board.csv": 259119,
    "dog.csv": 133852,
    "dolphin.csv": 101270,
    "donut.csv": 127280,
    "door.csv": 106127,
    "dragon.csv": 106206,
    "dresser.csv": 93187,
    "drill.csv": 106434,
    "drums.csv": 106936,
    "duck.csv": 105337,
    "dumbbell.csv": 127488,
    "ear.csv": 107295,
    "elbow.csv": 106946,
    "elephant.csv": 96716,
    "envelope.csv": 120472,
    "eraser.csv": 88229,
    "eye.csv": 111422,
    "eyeglasses.csv": 194804,
    "face.csv": 139237,
    "fan.csv": 105818,
    "feather.csv": 104013,
    "fence.csv": 115025,
    "finger.csv": 146017,
    "fire_hydrant.csv": 106916,
    "fireplace.csv": 125020,
    "firetruck.csv": 189791,
    "fish.csv": 116993,
    "flamingo.csv": 106924,
    "flashlight.csv": 208710,
    "flip_flops.csv": 103405,
    "floor_lamp.csv": 135777,
    "flower.csv": 129260,
    "flying_saucer.csv": 121526,
    "foot.csv": 172261,
    "fork.csv": 109697,
    "frog.csv": 136102,
    "frying_pan.csv": 107653,
    "garden.csv": 130865,
    "garden_hose.csv": 106551,
    "giraffe.csv": 112752,
    "goatee.csv": 159246,
    "golf_club.csv": 164041,
    "grapes.csv": 134134,
    "grass.csv": 110223,
    "guitar.csv": 102481,
    "hamburger.csv": 114329,
    "hammer.csv": 102068,
    "hand.csv": 260584,
    "harp.csv": 254320,
    "hat.csv": 191651,
    "headphones.csv": 105119,
    "hedgehog.csv": 94621,
    "helicopter.csv": 129323,
    "helmet.csv": 105725,
    "hexagon.csv": 126212,
    "hockey_puck.csv": 172443,
    "hockey_stick.csv": 114365,
    "horse.csv": 147547,
    "hospital.csv": 136803,
    "hot_air_balloon.csv": 111800,
    "hot_dog.csv": 151304,
    "hot_tub.csv": 90199,
    "hourglass.csv": 119293,
    "house.csv": 122595,
    "house_plant.csv": 107242,
    "hurricane.csv": 120707,
    "ice_cream.csv": 110131,
    "jacket.csv": 185135,
    "jail.csv": 105934,
    "kangaroo.csv": 143747,
    "keyboard.csv": 156983,
    "key.csv": 130356,
    "knee.csv": 236413,
    "ladder.csv": 111909,
    "lantern.csv": 119389,
    "laptop.csv": 230366,
    "leaf.csv": 107124,
    "leg.csv": 101213,
    "light_bulb.csv": 105750,
    "lighthouse.csv": 130255,
    "lightning.csv": 132380,
    "line.csv": 131061,
    "lion.csv": 90871,
    "lipstick.csv": 111752,
    "lobster.csv": 109817,
    "lollipop.csv": 114782,
    "mailbox.csv": 115072,
    "map.csv": 102861,
    "marker.csv": 287910,
    "matches.csv": 113577,
    "megaphone.csv": 112287,
    "mermaid.csv": 149497,
    "microphone.csv": 104070,
    "microwave.csv": 114413,
    "monkey.csv": 97391,
    "moon.csv": 105384,
    "mosquito.csv": 107125,
    "motorbike.csv": 139879,
    "mountain.csv": 113801,
    "mouse.csv": 156901,
    "moustache.csv": 154450,
    "mouth.csv": 116650,
    "mug.csv": 131433,
    "mushroom.csv": 127329,
    "nail.csv": 128067,
    "necklace.csv": 103557,
    "nose.csv": 173232,
    "ocean.csv": 116896,
    "octagon.csv": 138439,
    "octopus.csv": 134000,
    "onion.csv": 113472,
    "oven.csv": 183203,
    "owl.csv": 138963,
    "paintbrush.csv": 156202,
    "paint_can.csv": 101525,
    "palm_tree.csv": 107447,
    "panda.csv": 86499,
    "pants.csv": 128728,
    "paper_clip.csv": 111828,
    "parachute.csv": 110955,
    "parrot.csv": 154790,
    "passport.csv": 119778,
    "peanut.csv": 112421,
    "pear.csv": 104685,
    "peas.csv": 131094,
    "pencil.csv": 101025,
    "penguin.csv": 222697,
    "piano.csv": 104250,
    "pickup_truck.csv": 106936,
    "picture_frame.csv": 108080,
    "pig.csv": 155968,
    "pillow.csv": 102402,
    "pineapple.csv": 112175,
    "pizza.csv": 114877,
    "pliers.csv": 141937,
    "police_car.csv": 113396,
    "pond.csv": 101507,
    "pool.csv": 118378,
    "popsicle.csv": 112072,
    "postcard.csv": 108339,
    "potato.csv": 297953,
    "power_outlet.csv": 138828,
    "purse.csv": 103512,
    "rabbit.csv": 124720,
    "raccoon.csv": 92737,
    "radio.csv": 119060,
    "rainbow.csv": 115365,
    "rain.csv": 121008,
    "rake.csv": 124118,
    "remote_control.csv": 102883,
    "rhinoceros.csv": 157726,
    "river.csv": 117319,
    "roller_coaster.csv": 113177,
    "rollerskates.csv": 105308,
    "sailboat.csv": 122532,
    "sandwich.csv": 107700,
    "saw.csv": 103405,
    "saxophone.csv": 100366,
    "school_bus.csv": 106573,
    "scissors.csv": 102245,
    "scorpion.csv": 135021,
    "screwdriver.csv": 93647,
    "sea_turtle.csv": 101008,
    "see_saw.csv": 110517,
    "shark.csv": 109987,
    "sheep.csv": 108223,
    "shoe.csv": 106144,
    "shorts.csv": 111316,
    "shovel.csv": 102942,
    "sink.csv": 177500,
    "skateboard.csv": 115802,
    "skull.csv": 109804,
    "skyscraper.csv": 153008,
    "sleeping_bag.csv": 92463,
    "smiley_face.csv": 109898,
    "snail.csv": 118910,
    "snake.csv": 105850,
    "snorkel.csv": 124070,
    "snowflake.csv": 103589,
    "snowman.csv": 314440,
    "soccer_ball.csv": 109768,
    "sock.csv": 179562,
    "speedboat.csv": 157831,
    "spider.csv": 178828,
    "spoon.csv": 110325,
    "spreadsheet.csv": 139560,
    "square.csv": 110876,
    "squiggle.csv": 99899,
    "squirrel.csv": 126326,
    "stairs.csv": 113240,
    "star.csv": 123171,
    "steak.csv": 100177,
    "stereo.csv": 100606,
    "stethoscope.csv": 123295,
    "stitches.csv": 104865,
    "stop_sign.csv": 105167,
    "stove.csv": 86488,
    "strawberry.csv": 108849,
    "streetlight.csv": 107862,
    "string_bean.csv": 88996,
    "submarine.csv": 105676,
    "suitcase.csv": 111640,
    "sun.csv": 119962,
    "swan.csv": 124237,
    "sweater.csv": 106401,
    "swing_set.csv": 106676,
    "sword.csv": 107811,
    "table.csv": 112546,
    "teapot.csv": 110655,
    "teddy-bear.csv": 148791,
    "telephone.csv": 102597,
    "television.csv": 108129,
    "tennis_racquet.csv": 200184,
    "tent.csv": 117187,
    "The_Eiffel_Tower.csv": 120727,
    "The_Great_Wall_of_China.csv": 162226,
    "The_Mona_Lisa.csv": 102353,
    "tiger.csv": 101137,
    "toaster.csv": 94198,
    "toe.csv": 123183,
    "toilet.csv": 115449,
    "toothbrush.csv": 109036,
    "tooth.csv": 105546,
    "toothpaste.csv": 114380,
    "tornado.csv": 128847,
    "tractor.csv": 96016,
    "traffic_light.csv": 110153,
    "train.csv": 104626,
    "tree.csv": 128966,
    "triangle.csv": 110728,
    "trombone.csv": 154008,
    "truck.csv": 113249,
    "trumpet.csv": 138931,
    "t-shirt.csv": 111214,
    "umbrella.csv": 110813,
    "underwear.csv": 108476,
    "van.csv": 135315,
    "vase.csv": 113698,
    "violin.csv": 186372,
    "washing_machine.csv": 106878,
    "watermelon.csv": 115107,
    "waterslide.csv": 154592,
    "whale.csv": 99696,
    "wheel.csv": 120284,
    "windmill.csv": 104778,
    "wine_bottle.csv": 111732,
    "wine_glass.csv": 115445,
    "wristwatch.csv": 132025,
    "yoga.csv": 249290,
    "zebra.csv": 118335,
    "zigzag.csv": 106925,
}


def perturb(lines, d):
    """
    """
    output_lines = []

    for xs, ys in lines:
        xs = [x + random.randint(-d, d) for x in xs]
        ys = [y + random.randint(-d, d) for y in ys]

        output_lines.append((xs, ys))

    # NOTE: rotate

    return output_lines


def normalize(lines, image_size):
    """
    """
    def extremum(ls, idx, fun):
        """
        """
        for i, points in enumerate(ls):
            m = fun(points[idx])

            n = m if i == 0 else fun(n, m)

        return n

    output_lines = []

    min_x, min_y = extremum(lines, 0, min), extremum(lines, 1, min)
    max_x, max_y = extremum(lines, 0, max), extremum(lines, 1, max)

    # NOTE: scale to fix image_size
    s = max(max_x - min_x, max_y - min_y)
    t = image_size - 1

    for xs, ys in lines:
        xs = [(x - min_x) * t // s for x in xs]
        ys = [(y - min_y) * t // s for y in ys]

        output_lines.append((xs, ys))

    lines, output_lines = output_lines, []

    # NOTE: move to center
    tx = (t - extremum(lines, 0, max)) // 2
    ty = (t - extremum(lines, 1, max)) // 2

    for xs, ys in lines:
        xs = [x + tx for x in xs]
        ys = [y + ty for y in ys]

        output_lines.append((xs, ys))

    return output_lines


def lines_to_image(image_size, lines, add_perturbation=False):
    """
    """
    if perturb:
        lines = perturb(lines, 4)

    lines = normalize(lines, image_size)

    image = np.zeros((image_size, image_size), dtype=np.uint8)

    for xs, ys in lines:
        for i in range(1, len(xs)):
            rr, cc = skimage.draw.line(ys[i-1], xs[i-1], ys[i], xs[i])

            image[rr, cc] = 255

    return image


def int64_feature(v):
    """
    create a feature which contains a 64-bits integer
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def image_feature(image):
    """
    create a feature which contains 32-bits floats in binary format.
    """
    image = image.astype(np.uint8).tostring()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))


def load_block(args):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    csv_path, num_csv_lines, block = args

    # NOTE: find label
    basename = os.path.splitext(csv_path)[0]
    basename = os.path.basename(basename)
    basename = basename.replace(' ', '_')

    label = dataset.label_to_index[basename]

    # NOTE: find range of block
    if block < 0:
        # NOTE: load all
        negative, head, tail = True, -1, -1
    else:
        offset = 10000 * block % num_csv_lines

        if offset + 10000 > num_csv_lines:
            negative, head, tail = True, offset + 10000 - num_csv_lines, offset
        else:
            negative, head, tail = False, offset, offset + 10000

    # NOTE: start to purturb after 1 episode
    add_purturbation = 10000 * (block + 1) > num_csv_lines

    preprocessed_data = []

    # NOTE: load positive images
    with open(csv_path, newline='') as csv_file:
        draws = csv.reader(csv_file, delimiter=',')

        for index, draw in enumerate(draws):
            if (index >= head and index < tail) == negative:
                continue

            lines = json.loads(draw[1])
            image = lines_to_image(FLAGS.image_size, lines, add_purturbation)

            preprocessed_data.append((image, label))

            if len(preprocessed_data) % 100 == 0:
                print('{} / {}'.format(len(preprocessed_data), index))

            if len(preprocessed_data) >= 10000:
                break

    return preprocessed_data


def preprocess_train():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    csv_names = os.listdir(FLAGS.source_dir)

    args = []

    for name in csv_names:
        path = os.path.join(FLAGS.source_dir, name)

        num_lines = num_train_csv_lines[name]

        block = FLAGS.block

        args.append((path, num_lines, block))

    with multiprocessing.dummy.Pool(340) as pool:
        class_blocks = pool.map(load_block, args, 1)

    with tf.python_io.TFRecordWriter(FLAGS.result_tfr) as writer:
        for i in range(10000):
            for blocks in class_blocks:
                image, label = blocks[i]

                feature = {
                    'image': image_feature(image),
                    'label': int64_feature(label),
                }

                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))

                writer.write(example.SerializeToString())


def preprocess_test():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    preprocessed_data = []

    # NOTE: preprocess all image
    with open(FLAGS.source_csv, newline='') as csv_file:
        draws = csv.reader(csv_file, delimiter=',')

        # NOTE: skip header
        next(draws)

        for draw in draws:
            lines = json.loads(draw[2])
            image = lines_to_image(FLAGS.image_size, lines, FLAGS.block > 0)

            preprocessed_data.append(image)

    # NOTE: output to 2 tfrecord
    with tf.python_io.TFRecordWriter(FLAGS.result_tfr) as writer:
        for image in preprocessed_data:
            feature = {'image': image_feature(image)}

            example = tf.train.Example(
                features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.source_dir is not None and tf.gfile.Exists(FLAGS.source_dir):
        preprocess_train()

    if FLAGS.source_csv is not None and tf.gfile.Exists(FLAGS.source_csv):
        preprocess_test()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('source_dir', None, '')

    tf.app.flags.DEFINE_string('source_csv', None, '')
    tf.app.flags.DEFINE_string('result_tfr', None, '')

    tf.app.flags.DEFINE_integer('block', 0, '')

    tf.app.flags.DEFINE_integer('image_size', 28, '')

    tf.app.run()

