require './NeuralNetwork'

net = NeuralNetwork.new(256, 22, 10, 0.2, 0.4)

matrix = IO.readlines('numbers.data').map { |line| line.split.map(&:to_i) }
training_data = []

matrix.each do |row|
  training_data << [row[0..255], row[-10..-1]]
end

File.open('training_results.txt', 'w') do |file|
  20.times do |iter|
    pp net.train(training_data.sample(40), iter, file)
  end
end


net.test(training_data.sample(100))

