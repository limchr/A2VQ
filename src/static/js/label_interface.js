// <!--
// #!/usr/bin/env python
// #
// # Copyright (C) 2019
// # Christian Limberg
// # Centre of Excellence Cognitive Interaction Technology (CITEC)
// # Bielefeld University
// #
// #
// # Redistribution and use in source and binary forms, with or without modification,
// # are permitted provided that the following conditions are met:
// #
// # 1. Redistributions of source code must retain the above copyright notice,
// # this list of conditions and the following disclaimer.
// #
// # 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
// # and the following disclaimer in the documentation and/or other materials provided with the distribution.
// #
// # 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
// # products derived from this software without specific prior written permission.
// #
// # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// # INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// # WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
// # THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// #
// -->

var zoom;





document.addEventListener("DOMContentLoaded", function(event) {

    // get div width for setting up svg width to that
    w = document.getElementById('canvas_container').offsetWidth;

    //
    //set up d3
    // 
    xScale = d3.scaleLinear().range([0, w]).domain([0, 1]).nice();
    yScale = d3.scaleLinear().range([0, h]).domain([0, 1]).nice();

    //append svg element
    svg = d3.select('#canvas_container').append('svg')
    .attr("id", 'svg_container')
    .attr("width", w)
    .attr("height", h)
    .attr("style", "outline: 3px solid black")
    .on("mouseup", mouseup)
    .on("contextmenu", function(event) {
        handle_right_click(event);
    });

    // greate zoom group that is transformed and paned using zoom and a2vq querying
    zoom_group = svg.append("g").attr("id","zoom_group").attr("transform", "translate(0,0)");


    // load thumbnails from variable transmitted from backend in html code
    thumbs = zoom_group
    .selectAll("image")
    .data(Object.values(samples))
    .enter()
    .append("svg:image")
    .attr("x", d => xScale(d.embedding[0])-s/2)
    .attr("y", d => yScale(d.embedding[1])-s/2)
    .attr("id", d => d.id)
    .attr("label", d => d.label)
    .attr("href", function(d) {return 'data:image/png;base64,'+d.thumb})
    .attr("height", s)
    .attr("width", s);

//// this would be center the samples correctly but idk why there are bugs related to user input
//   bbox = zoom_group.node().getBBox();
//   vx = bbox.x;		// container x co-ordinate
//   vy = bbox.y;		// container y co-ordinate
//   vw = bbox.width;	// container width
//   vh = bbox.height;	// container height
//   defaultView = "" + vx + " " + vy + " " + vw + " " + vh;
//   svg
// 	.attr("viewBox", defaultView);



    // setting up zoom function (this is used at mouse zoom and a2vq querying function)
    zoom = d3.zoom()
      .extent([[0, 0], [w, h]])
      .scaleExtent([1, 50])
      .on("zoom", zoomed);
    svg.call(zoom);


    // color drop down menu in class specific colors
    opts = Array.from(document.querySelector("#label_name").options);
    for(var i=0;i<opts.length;i=i+1) {
        opts[i].style.color = colors.geti(i);
    }


    //update outline of samples
    update_selections();


    //register callbacks for user input (buttons)
    $('#label_button').click(function(){label_selected();});
    $('#remove_button').click(function(){selections = []; update_selections();});
    $('#new_class_button').click(function(){window.location.replace('/add_class_'+$('#class_label').val());});
    $('#querying_button').click(function(){a2vq_querying();});


}); // document loaded


//this function redraws outlines of samples (either class colors, or if they are selected (red) or neither)
function update_selections(){
    var imgs = Array.from(document.getElementById('svg_container').querySelectorAll("image"));

    for(i=0;i<imgs.length;i++) {
      if(imgs[i].id in selections) { // sample is selected and has to be marked in red
       imgs[i].style.outline = '1px solid red';
      } else { //sample is not selected, mark in class specific color
        var label = Object.values(samples)[i].label;
        if(class_names.includes(label)) {
          col_i = class_names.indexOf(label);
          imgs[i].style.outline = '0.001em solid '+colors.geti(col_i);
        } else { //unlabeled or unknown label (not in class_names list)
          imgs[i].style.outline = 'none';
        }
      }
    }

}


// label the selected samples and send them over asyncronous javascript to the backend, remove all selections and update classes
function label_selected(){
  class_name = $('#label_name').val();
  if(selections.length == 0){
  return;
  }

    disable_buttons();

   $.ajax({
          type: "POST",
          cache: false,
          data:{'label': class_name, 'ids': selections, 'num_rects': -1},
          url: '/add_labels',
          dataType: "json",
          success: function(data) {
            if(data["success"] == 'false'){
              window.open(data["open_page"],"_self");
            }
              console.log(data);
          },
          error: function(jqXHR) {
              alert("error: " + jqXHR.status);
              console.log(jqXHR);
          }
      });

        for (var i = 0; i<selections.length; ++i){
            samples[selections[i]].label = class_name;
        }
        selections = [];
        update_selections();

        enable_buttons();
        query_new_views = true;

}

// a2vq querying variables
var next_view = 0;
var views = null;
var scores =null;
var query_new_views = true;
var view_size = null;
var overlap = null;

// get the actual most desireable views from the backend (if it is providing posterior probabilities) and pan-and-zoom to this areas
function a2vq_querying(){
    if(query_new_views) { // new labels available - need to request new views from backend
    	    disable_buttons();

	   $.ajax({
			  type: "POST",
			  cache: false,
			  async: false,
			  data:{},
			  url: '/a2vq',
			  dataType: "json",
			  success: data => {
				views = data['views'];
				scores = data['scores'];
				view_size = parseFloat(data['view_size']);
				overlap = parseFloat(data['overlap']);
			    query_new_views = false;
			    next_view = 0;
			  },
			  error: function(jqXHR) {
				  alert("error: " + jqXHR.status);
				  console.log(jqXHR);
			  }
		  });
		      enable_buttons()
      }


	view = views[next_view];
	scale = 1/view_size;
	next_view = next_view+1;

	//zoom out, pan and zoom in

	//calculation from https://bl.ocks.org/catherinekerr/b3227f16cebc8dd8beee461a945fb323

	offset = getCumulativeOffset(document.getElementById('canvas_container'));
	bb = zoom_group.node().getBBox();

	bx = (-1*(view[0]+(view_size/2))*scale)*w + w/2 + bb.x;
	by = (-1*(view[1]+(view_size/2))*scale)*h + h/2 + bb.y;


	svg.transition()
	.duration(1000)
	.call(
	zoom.transform, d3.zoomIdentity.translate(0,0)
	.scale(1)
	.translate(0,0)
	).transition()
	.duration(2000).call(
	zoom.transform, d3.zoomIdentity.translate(bx,by)
	.scale(scale)
	);


}

// disable buttons if waiting for a response from server
function disable_buttons() {
      $('#label_button').addClass('disabled');
  $('#next_view_button').addClass('disabled');

  $('#label_button').addClass('btn-warning');
  $('#next_view_button').addClass('btn-warning');
  $('#next_view_link').attr('href', '/');

}

//make all buttons active again
function enable_buttons() {
                    $('#label_button').removeClass('disabled');
                $('#next_view_button').removeClass('disabled');
                $('#label_button').removeClass('btn-warning');
                $('#next_view_button').removeClass('btn-warning');
                $('#label_button').addClass('btn-primary');
                $('#next_view_button').addClass('btn-success');
                $('#next_view_link').attr('href', '/');
}

// zoom function (manual caclulations where needed because the thumbnail size should stay the same at all zoom level for resolving ambiguities regarding to overlapping samples)
function zoomed({transform}) {
	zoom_group.attr("transform", transform);
    var zoom_scale = d3.zoomTransform(svg.node()).k;
    var img_size = s/zoom_scale;
    var margin = (s - img_size)/2;

    var imgs = Array.from(document.getElementById('svg_container').querySelectorAll("image"));

     for(i=0;i<imgs.length;i++) {
        imgs[i].x.baseVal.value = xScale(Object.values(samples)[i].embedding[0]) - s/2 + margin;
        imgs[i].y.baseVal.value = yScale(Object.values(samples)[i].embedding[1]) - s/2 + margin;
     }

    svg.selectAll('image')
    .attr('width',img_size)
    .attr('height',img_size);
}


// transformation of mouse coordinates to actual svg coordinates (might not be completely correct)
function transform_point(point) {
		var svg_dom = document.getElementById('svg_container');
		var zoom_dom = document.getElementById('zoom_group');

		cumulativeOffset = getCumulativeOffset(document.getElementById('canvas_container'));
        bb = zoom_group.node().getBBox();

		svgpoint = svg_dom.createSVGPoint();
		svgpoint.x = point[0] + cumulativeOffset.left - bb.x/2;
		svgpoint.y = point[1] + cumulativeOffset.top;
		var icmt = zoom_dom.getScreenCTM().inverse();
		var transformed_pt = svgpoint.matrixTransform(icmt);
		//console.log('x='+transformed_pt.x+' y='+transformed_pt.y);


		return transformed_pt;

	}

// selection is drawn (therefore a rect in svg is created, which is updated in mousemove)
function handle_right_click(event) {
    console.log('right click');
		event.preventDefault();
		var mousepoint = d3.pointer(event);
        origin = transform_point(mousepoint);

		rect = zoom_group.append("rect")
			.attr("x", origin.x)
			.attr("y", origin.y)
			.attr("height", 0)
			.attr("width", 0)
            .attr("style", "fill:rgba(150,150,150,0.3);stroke-width:1;stroke:rgb(0,0,0)");
		svg.on("mousemove", mousemove);
}

// updates selection rect to new mouse coordinates
function mousemove(event) {
    var m = d3.pointer(event);
    var t = transform_point(m);
	var x_min = Math.min(origin.x, t.x);
	var x_max = Math.max(origin.x, t.x);
	var y_min = Math.min(origin.y, t.y);
	var y_max = Math.max(origin.y, t.y);

    rect.attr("x", x_min)
        .attr("y", y_min)
        .attr("width", x_max-x_min)
        .attr("height", y_max-y_min)
        ;
}


// right mouse button was released - mark the underlying rect samples and remove rectangle
function mouseup() {
    console.log('mouse up');
    svg.on("mousemove", null);

    xs = parseInt(rect.attr('x'));
	ys = parseInt(rect.attr('y'));
	ws = parseInt(rect.attr('width'));
	hs = parseInt(rect.attr('height'));

    var imgs = Array.from(document.getElementById('svg_container').querySelectorAll("image"));


     for(i=0;i<imgs.length;i++) {

        xi = imgs[i].x.baseVal.value + imgs[i].width.baseVal.value/2;
        yi = imgs[i].y.baseVal.value + imgs[i].height.baseVal.value/2;

        if(selection_intersects([xs,xs+ws,ys,ys+hs],[xi,yi])) {
        	imgs[i].style.outline = '1px solid red';
        	var new_id = parseInt(imgs[i].getAttribute('id'));
        	if (!selections.includes(new_id)){
        	    selections.push(new_id);
        	}
        }
     }
     zoom_group.selectAll('rect').remove();
}


// collision detection of rectangle and single point
function selection_intersects(selection, point) {
  return point[0] >= selection[0] && point[0] <= selection[1] && point[1] >= selection[2] && point[1] <= selection[3];
}

// get the cumulative offset of the svg element to the whole page by accumulating parent offsets
function getCumulativeOffset(element) {
    var top = 0, left = 0;
    do {
        top += element.offsetTop  || 0;
        left += element.offsetLeft || 0;
        element = element.offsetParent;
    } while(element);

    return {
        top: top,
        left: left
    };
}

