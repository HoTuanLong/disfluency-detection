from libs import *
from api import Disfluency

model = Disfluency("ckps/Disfluency/word")

examples = [
    "cho tôi xem giá vé cho các chuyến bay từ côn đảo đến đồng hới vào ngày mùng 3 tháng 7 năm 2021 à không xin lỗi ý tôi là ngày mùng 1 tháng 7 năm 2021",
    "có sân bay í lộn hãng hàng không nào có các chuyến bay từ điện biên phủ đến quảng ninh à chính xác là đến quy nhơn khởi hành trước 6 giờ 30 phút sáng không",
    "cho tôi loại máy bay mà thôi tôi đang tìm một chuyến bay từ hà nội đến amsterdam có trạm dừng ở bang bangkok và hy vọng là chuyến bay có phục vụ bữa tối làm sao tôi có thể tìm ra nó",
    "đà nẵng đến ờ hồ chí minh í lộn đến cà mau",
    "cho tôi biết về phương tiện giao thông đường bộ giữa sân bay quốc tế tân sơn nhất à nhầm vân đồn và hạ long",
]

demo = gr.Interface(
    title = "Vietnamese Disfluency Detecion",
    inputs = gr.Textbox(label='Text', placeholder="Type your sentence..."),
    outputs = gr.JSON(label = "Entities"), 
    fn = model.prediction, 
    examples=examples
)
demo.launch()
