from occupancy.vision.zones import Detection, ZoneRect, anchor_point, clamp_rect, count_by_zone

LEFT = ZoneRect(1, "Левая", 0.0, 0.0, 0.5, 1.0)
RIGHT = ZoneRect(2, "Правая", 0.5, 0.0, 1.0, 1.0)


def test_counts_people_by_center():
    detections = [Detection(10, 10, 50, 90), Detection(60, 10, 90, 90), Detection(70, 20, 100, 80)]
    assert count_by_zone(detections, [LEFT, RIGHT], 100, 100) == {1: 1, 2: 2}


def test_person_outside_all_zones_is_not_counted():
    top = ZoneRect(1, "Верх", 0.0, 0.0, 1.0, 0.3)
    assert count_by_zone([Detection(10, 60, 30, 100)], [top], 100, 100) == {1: 0}


def test_first_matching_zone_wins():
    whole = ZoneRect(3, "Всё", 0.0, 0.0, 1.0, 1.0)
    assert count_by_zone([Detection(10, 10, 20, 20)], [LEFT, whole], 100, 100) == {1: 1, 3: 0}


def test_bottom_anchor_moves_person_to_lower_zone():
    upper = ZoneRect(1, "Верх", 0.0, 0.0, 1.0, 0.5)
    lower = ZoneRect(2, "Низ", 0.0, 0.5, 1.0, 1.0)
    person = Detection(40, 10, 60, 80)  # центр в верхней половине, ноги в нижней
    assert anchor_point(person, "center") == (50, 45)
    assert count_by_zone([person], [upper, lower], 100, 100, anchor="center") == {1: 1, 2: 0}
    assert count_by_zone([person], [upper, lower], 100, 100, anchor="bottom") == {1: 0, 2: 1}


def test_zone_pixels_follow_frame_size():
    assert RIGHT.pixels(1280, 720) == (640, 0, 1280, 720)
    assert RIGHT.pixels(1920, 1080) == (960, 0, 1920, 1080)


def test_clamp_rect_orders_and_limits_coordinates():
    assert clamp_rect(0.9, 1.4, 0.2, -0.3) == [0.2, 0.0, 0.9, 1.0]
