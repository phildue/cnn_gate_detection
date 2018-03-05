# thanks to https://github.com/joycex99/tiny-yolo-keras





# def aug_img(train_instance):
#     path = train_instance['filename']
#     all_obj = copy.deepcopy(train_instance['object'][:])
#     img = cv2.imread(img_dir + path + ".JPEG")
#     h, w, c = img.shape
#
#     # scale the image
#     scale = np.random.uniform() / 10. + 1.
#     img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
#
#     # translate the image
#     max_offx = (scale - 1.) * w
#     max_offy = (scale - 1.) * h
#     offx = int(np.random.uniform() * max_offx)
#     offy = int(np.random.uniform() * max_offy)
#     img = img[offy: (offy + h), offx: (offx + w)]
#
#     # flip the image
#     flip = np.random.binomial(1, .5)
#     if flip > 0.5: img = cv2.flip(img, 1)
#
#     # re-color
#     t = [np.random.uniform()]
#     t += [np.random.uniform()]
#     t += [np.random.uniform()]
#     t = np.array(t)
#
#     img = img * (1 + t)
#     img = img / (255. * 2.)
#
#     # resize the image to standard size
#     img = cv2.resize(img, (NORM_H, NORM_W))
#     img = img[:, :, ::-1]
#
#     # fix object's position and size
#     for obj in all_obj:
#         for attr in ['xmin', 'xmax']:
#             obj[attr] = int(obj[attr] * scale - offx)
#             obj[attr] = int(obj[attr] * float(NORM_W) / w)
#             obj[attr] = max(min(obj[attr], NORM_W), 0)
#
#         for attr in ['ymin', 'ymax']:
#             obj[attr] = int(obj[attr] * scale - offy)
#             obj[attr] = int(obj[attr] * float(NORM_H) / h)
#             obj[attr] = max(min(obj[attr], NORM_H), 0)
#
#         if flip > 0.5:
#             xmin = obj['xmin']
#             obj['xmin'] = NORM_W - obj['xmax']
#             obj['xmax'] = NORM_W - xmin
#
#     return img, all_obj
