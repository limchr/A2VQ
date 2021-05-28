
function px2coord(px_vec) {
    return [(px_vec[0]-orig[0])/grid_w,(orig[1]-px_vec[1])/grid_w];
}

function coord2px(coord_vec) {
    return [(coord_vec[0] * grid_w) + orig[0], orig[1]-(coord_vec[1] * grid_w)];
}




function pad(num, size) {
    var s = num+"";
    while (s.length < size) s = "0" + s;
    return s;
}

function get_mouse_xy(can, e) {
  var element = can, offsetX = 0, offsetY = 0, mx, my;
  if (element.offsetParent !== undefined) {
    do {
      offsetX += element.offsetLeft;
      offsetY += element.offsetTop;
    } while ((element = element.offsetParent));
  }
  mx = e.pageX - offsetX;
  my = e.pageY - offsetY;
  return {x: mx, y: my};
}

function selection_intersects(selection, point) {
  return point[0] >= selection[0] && point[0] <= selection[1] && point[1] >= selection[2] && point[1] <= selection[3];
}

