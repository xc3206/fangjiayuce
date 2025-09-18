// 房屋数据生成器
export interface HouseData {
  area: number; // 面积 (平方米)
  bedrooms: number; // 卧室数量
  bathrooms: number; // 浴室数量
  age: number; // 房龄 (年)
  floor: number; // 楼层
  hasParking: number; // 是否有停车位 (0/1)
  hasGarden: number; // 是否有花园 (0/1)
  distanceToCenter: number; // 距离市中心距离 (公里)
  price: number; // 价格 (万元)
}

// 生成模拟房屋数据
export function generateHouseData(count: number): HouseData[] {
  const data: HouseData[] = [];
  
  for (let i = 0; i < count; i++) {
    // 基础特征
    const area = Math.random() * 150 + 50; // 50-200平方米
    const bedrooms = Math.floor(Math.random() * 4) + 1; // 1-4个卧室
    const bathrooms = Math.floor(Math.random() * 3) + 1; // 1-3个浴室
    const age = Math.random() * 30; // 0-30年房龄
    const floor = Math.floor(Math.random() * 20) + 1; // 1-20楼
    const hasParking = Math.random() > 0.4 ? 1 : 0; // 60%概率有停车位
    const hasGarden = Math.random() > 0.7 ? 1 : 0; // 30%概率有花园
    const distanceToCenter = Math.random() * 25 + 2; // 2-27公里
    
    // 根据特征计算价格（加入一些噪声）
    let price = area * 0.8 + // 面积影响
                bedrooms * 15 + // 卧室数影响
                bathrooms * 10 + // 浴室数影响
                (30 - age) * 0.5 + // 房龄影响（越新越贵）
                hasParking * 20 + // 停车位加价
                hasGarden * 25 + // 花园加价
                Math.max(0, (15 - distanceToCenter)) * 2; // 距离市中心越近越贵
    
    // 添加随机噪声
    price += (Math.random() - 0.5) * 40;
    price = Math.max(30, price); // 最低30万
    
    data.push({
      area: Math.round(area * 10) / 10,
      bedrooms,
      bathrooms,
      age: Math.round(age * 10) / 10,
      floor,
      hasParking,
      hasGarden,
      distanceToCenter: Math.round(distanceToCenter * 10) / 10,
      price: Math.round(price * 10) / 10
    });
  }
  
  return data;
}