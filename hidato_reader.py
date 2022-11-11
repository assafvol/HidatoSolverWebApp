from hidato_reader_utils import *
from custom_solver import *


def read_and_solve_hidato(img,  model, on_original_image=True, heightImg=500, widthImg=500, debug=False):
    # resize to square and convert to binary image
    img = cv2.resize(img, (widthImg, heightImg))
    imgThreshold = pre_process(img)

    # find contours
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        output = img.copy()
        output = cv2.drawContours(output, contours, -1, (0,255,0), 3)
        cv2.imshow('Contours', output)
        cv2.waitKey(0)

    # find biggest contour
    biggest_contour, biggest_polygon, perimeter = find_biggest_contour(contours)
    biggest_polygon = fit_contour_to_a_possible_hidato_size(biggest_contour, biggest_polygon, perimeter)

    if debug:
        output = img.copy()
        output = cv2.drawContours(output, biggest_polygon, -1, (0, 255, 0), 10)
        cv2.imshow('Biggest polygon', output)
        cv2.waitKey(0)

    # find the four corners of the hidato polygon
    polygon = np.squeeze(biggest_polygon)
    four_corners = get_four_hidato_corners(polygon)

    if debug:
        output = img.copy()
        output = cv2.drawContours(output, four_corners.reshape(4,1,2), -1, (0, 255, 0), 10)
        cv2.imshow('Four corners', output)
        cv2.waitKey(0)

    # perspective transformation of image, hidato polygon, and polygon four corners
    imgWarpColored, polygon_warped, corners_warped, matrix = transform_perspective(polygon, four_corners, img,
                                                                                   (widthImg, heightImg))

    if debug:
        output = imgWarpColored.copy()
        output = cv2.drawContours(output, np.expand_dims(np.array(polygon_warped), axis=1), -1, (0, 255, 0), 3)
        output = cv2.drawContours(output, corners_warped.reshape(4, 1, 2), -1, (0, 0, 255), 10)

        cv2.imshow('Image warped with four corners', output)
        cv2.waitKey(0)

    # find the lattice of hexagons (an array with the centers of all the hexagonal cells in the hidato
    lattice, h, w = find_hexagons(np.array(polygon_warped), corners_warped)

    if debug:
        output = imgWarpColored.copy()
        for i, row in enumerate(lattice):
            for j, centroid in enumerate(row):
                output = cv2.circle(output, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)
        cv2.imshow('Image warped with lattice centers', output)
        cv2.waitKey(0)

    # convert the warped img to binary
    imgWarpBinary = pre_process(imgWarpColored)
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

    # print(f"noise_size = {0.03 * w * h}")
    # with open("imWarpBinary.npy","wb") as f:
    #     np.save(f,imgWarpBinary)
    # with open("imWarpGray.npy","wb") as f:
    #     np.save(f,imgWarpGray)
        
    imgWarpBinary, imgWarpGray = remove_hexagonal_grid_and_noise(imgWarpBinary, imgWarpGray, noise_size=0.03 * w * h)

    if debug:
        output_binary = imgWarpBinary.copy()
        output_gray = imgWarpGray.copy()
        cv2.imshow('Binary image with background and noise removed', output_binary)
        cv2.imshow('Gray image with background and noise removed', output_gray)
        cv2.waitKey(0)

    boxes = get_boxes(lattice, h, w, imgWarpBinary)
    boxes_gray = get_boxes(lattice, h, w, imgWarpGray)


    # Find digits inside boxes.
    digits = [[[] for col in row] for row in boxes]
    for i, row in enumerate(boxes):
        for j, box in enumerate(row):
            nb_components, _, stats, centroids = cv2.connectedComponentsWithStats(box, connectivity=4)
            for x, y, width, height, _ in sorted(stats[1:], key=lambda lst: lst[0]):
                digits[i][j].append(cv2.resize(crop_from_image(boxes_gray[i][j], (x, y, width, height)), (28, 28)))

    digit_images = []
    digit_indices = []
    for i, row in enumerate(digits):
        for j, digit_list in enumerate(row):
            if digit_list:
                for digit_img in digit_list:
                    digit_images.append(digit_img)
                    digit_indices.append([i, j])

    digit_images = np.expand_dims(np.array(digit_images), axis=3) / 255

    if debug:
        n = len(digit_images)
        nc = int(np.sqrt(n))+1
        fig, ax = plt.subplots(nrows=nc,ncols=nc, figsize=(15,15))
        for i,digit_image in enumerate(digit_images):
            i_row, i_col = i // nc, i % nc
            ax[i_row][i_col].imshow(digit_image)
        plt.tight_layout()
        
    digit_probas = model.predict(digit_images)

    final_lattice = [[[] for _ in row] for row in lattice]
    for cell_indices, digit_proba in zip(digit_indices, digit_probas):
        i, j = cell_indices
        max_proba = np.max(digit_proba)
        digit_pred = np.argmax(digit_proba)
        if max_proba >= 0.85:
            final_lattice[i][j].append(digit_pred)

    final_lattice = [[int(''.join(map(str, cell))) if cell else 0 for cell in row] for row in final_lattice]

    if debug:
        hidato = Hidato(final_lattice)
        hidato.plot(initial_cells_only=True)
        val = input("Press c to to continue or q to quit")
        while val not in ('c','q'):
            val = input("Invalid input, press c to to continue or q to quit")
        if val == 'q':
            return
    
    solution_lattice = solve_with_cache(final_lattice)

    if solution_lattice is None:
        raise Exception("Could not solve the hidato")

    if on_original_image:
        for i, row in enumerate(solution_lattice):
            for j, number in enumerate(row):
                if final_lattice[i][j] == 0:
                    # desired_width = 0.8*sum(digit[0] for digit in digits[i][j])
                    # font_scale = get_optimal_font_scale(str(number), desired_width)
                    font_scale = 1
                    imgWarpColored = put_text_centered(imgWarpColored, str(number), lattice[i][j], font_scale)

        inverseImgWarpColored = cv2.warpPerspective(imgWarpColored, matrix, (widthImg, heightImg), flags=cv2.WARP_INVERSE_MAP)
        inverseImgWarpGray = cv2.cvtColor(inverseImgWarpColored, cv2.COLOR_BGR2GRAY)

        final_image = np.where((inverseImgWarpGray == 0).reshape(500, 500, 1), img, inverseImgWarpColored)
        return final_image
    else:
        initial_cells = {(i, j) for i in range(len(final_lattice)) for j in range(len(final_lattice[i]))
                         if final_lattice[i][j] != 0}
        img = Hidato(lattice=solution_lattice, initial_cells=initial_cells).plot(return_as_image=True)
        return img
